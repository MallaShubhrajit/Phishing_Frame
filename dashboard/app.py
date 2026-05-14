"""
app.py — Streamlit Dashboard
Phishing Detection & Log Anomaly Detection Framework
Group 19-10 | SOA University ITER
Label convention: 0.0 = Phishing  |  1.0 = Legitimate

Detection approach (based on best GitHub phishing detection projects):
Layer 1: Dataset Lookup  — exact match in training data
Layer 2: Whitelist       — known safe domains
Layer 3: Score-Based     — weighted suspicious signal scoring
Layer 4: ML Model        — Random Forest fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import math
from urllib.parse import urlparse

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "Phishing Detection Dashboard",
    page_icon  = "shield",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Paths ─────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE, 'models',  'random_forest.pkl')
SCALER_PATH = os.path.join(BASE, 'models',  'scaler.pkl')
FEAT_PATH   = os.path.join(BASE, 'data', 'processed', 'feature_cols.npy')
DATA_PATH   = os.path.join(BASE, 'data', 'raw', 'url_features_cleaned.csv')
LOG_PATH    = os.path.join(BASE, 'data', 'processed', 'log_features_scored.csv')
RETRAIN_LOG = os.path.join(BASE, 'reports', 'retraining_log.json')
ROC_IMG     = os.path.join(BASE, 'reports', 'roc_curves.png')
CM_IMG      = os.path.join(BASE, 'reports', 'confusion_matrices.png')
FI_IMG      = os.path.join(BASE, 'reports', 'feature_importance.png')
RH_IMG      = os.path.join(BASE, 'reports', 'retraining_history.png')


# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feats  = np.load(FEAT_PATH, allow_pickle=True).tolist()
    return model, scaler, feats


@st.cache_resource
def load_url_lookup():
    """Load dataset URLs for direct lookup. Returns dict url->label."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, usecols=['URL', 'ClassLabel'])
        df = df.dropna(subset=['ClassLabel'])
        return dict(zip(df['URL'].str.strip(), df['ClassLabel']))
    return {}


model, scaler, feature_cols = load_models()
url_lookup = load_url_lookup()


# ── Known legitimate domains whitelist ────────────────────────────
KNOWN_LEGIT = {
    'google.com', 'github.com', 'microsoft.com', 'stackoverflow.com',
    'wikipedia.org', 'youtube.com', 'amazon.com', 'linkedin.com',
    'facebook.com', 'twitter.com', 'apple.com', 'instagram.com',
    'netflix.com', 'reddit.com', 'office.com', 'live.com',
    'outlook.com', 'yahoo.com', 'zoom.us', 'dropbox.com',
    'adobe.com', 'oracle.com', 'ibm.com', 'cisco.com',
    'anthropic.com', 'openai.com', 'twitch.tv', 'spotify.com',
    'paypal.com', 'ebay.com', 'twitter.com', 'x.com',
    'whatsapp.com', 'telegram.org', 'discord.com', 'slack.com',
    'notion.so', 'figma.com', 'canva.com', 'medium.com',
}

# ── Suspicious TLDs ───────────────────────────────────────────────
SUSPICIOUS_TLDS = {
    '.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.click',
    '.loan', '.work', '.date', '.racing', '.review', '.stream',
    '.gdn', '.bid', '.win', '.download', '.accountant', '.science',
    '.trade', '.webcam', '.faith', '.party', '.cricket', '.link',
    '.online', '.site', '.website', '.tech', '.info',
}

# ── Brand keywords ────────────────────────────────────────────────
BRAND_KEYWORDS = {
    'paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook',
    'instagram', 'netflix', 'bank', 'secure', 'verify', 'account',
    'login', 'signin', 'update', 'confirm', 'wallet', 'support',
    'helpdesk', 'ebay', 'wellsfargo', 'chase', 'citibank', 'hsbc',
    'barclays', 'dhl', 'fedex', 'ups', 'usps', 'irs', 'gov',
}


# ── URL Feature Extraction ────────────────────────────────────────
def extract_url_features(url):
    """Extract 16 features matching training columns."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path   = parsed.path

        freq = {}
        for c in url:
            freq[c] = freq.get(c, 0) + 1
        entropy = -sum(
            (f / len(url)) * math.log2(f / len(url))
            for f in freq.values()
        ) if len(url) > 0 else 0

        return {
            'url_length'               : len(url),
            'has_ip_address'           : 1 if re.match(
                                         r'\d+\.\d+\.\d+\.\d+',
                                         domain) else 0,
            'dot_count'                : url.count('.'),
            'https_flag'               : 1 if url.startswith('https') else 0,
            'url_entropy'              : round(entropy, 4),
            'token_count'              : len(re.split(r'[/\-_?=&.]', url)),
            'subdomain_count'          : max(len(domain.split('.')) - 2, 0),
            'query_param_count'        : url.count('='),
            'tld_length'               : len(domain.split('.')[-1])
                                         if '.' in domain else 0,
            'path_length'              : len(path),
            'has_hyphen_in_domain'     : 1 if '-' in domain else 0,
            'number_of_digits'         : sum(c.isdigit() for c in url),
            'tld_popularity'           : 1 if any(
                                         domain.endswith(t)
                                         for t in ['.com', '.org', '.edu',
                                                   '.gov', '.net', '.co.uk']
                                        ) else 0,
            'suspicious_file_extension': 1 if any(
                                          url.lower().endswith(e)
                                          for e in ['.exe', '.php', '.js',
                                                    '.sh', '.bat', '.zip']
                                         ) else 0,
            'domain_name_length'       : len(domain),
            'percentage_numeric_chars' : (
                sum(c.isdigit() for c in url) / max(len(url), 1)
            ),
        }
    except Exception:
        return None


def align_features(feat_dict):
    row = {col: feat_dict.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row])[feature_cols]


# ── Score-based phishing detection ───────────────────────────────
def compute_phishing_score(url, feat_dict):
    """
    Score-based detection from top GitHub phishing detection projects.
    Each suspicious signal adds points. Score >= threshold = phishing.
    Returns (score, max_score, signals_triggered)
    """
    parsed  = urlparse(url)
    domain  = parsed.netloc.lower()
    path    = parsed.path.lower()
    signals = []
    score   = 0

    # Signal 1: IP address in domain (very strong = 30 pts)
    if feat_dict.get('has_ip_address', 0) == 1:
        score += 30
        signals.append(('IP address in URL', 30))

    # Signal 2: @ symbol in URL (very strong = 25 pts)
    if '@' in url:
        score += 25
        signals.append(('@ symbol in URL', 25))

    # Signal 3: Suspicious TLD (strong = 20 pts)
    if any(domain.endswith(t) for t in SUSPICIOUS_TLDS):
        score += 20
        signals.append((f'Suspicious TLD', 20))

    # Signal 4: No HTTPS (moderate = 15 pts)
    if feat_dict.get('https_flag', 0) == 0:
        score += 15
        signals.append(('No HTTPS', 15))

    # Signal 5: Brand keyword in domain but not real domain (strong = 20 pts)
    brand_in_domain = any(b in domain for b in BRAND_KEYWORDS)
    is_real         = any(
        domain == d or domain.endswith('.' + d) for d in KNOWN_LEGIT
    )
    if brand_in_domain and not is_real:
        score += 20
        signals.append(('Brand impersonation in domain', 20))

    # Signal 6: Hyphen in domain (moderate = 10 pts)
    if feat_dict.get('has_hyphen_in_domain', 0) == 1:
        score += 10
        signals.append(('Hyphen in domain', 10))

    # Signal 7: Suspicious file extension (strong = 20 pts)
    if feat_dict.get('suspicious_file_extension', 0) == 1:
        score += 20
        signals.append(('Suspicious file extension', 20))

    # Signal 8: Many subdomains (moderate = 10 pts)
    if feat_dict.get('subdomain_count', 0) >= 3:
        score += 10
        signals.append(('Many subdomains', 10))

    # Signal 9: Very long URL (mild = 5 pts)
    if feat_dict.get('url_length', 0) > 75:
        score += 5
        signals.append(('Long URL', 5))

    # Signal 10: Many dots (mild = 5 pts)
    if feat_dict.get('dot_count', 0) >= 5:
        score += 5
        signals.append(('Many dots in URL', 5))

    # Signal 11: Phishing keywords in path (moderate = 15 pts)
    phish_path_keywords = [
        'verify', 'update', 'confirm', 'secure', 'login',
        'signin', 'account', 'banking', 'password', 'credential'
    ]
    if any(k in path for k in phish_path_keywords):
        score += 15
        signals.append(('Phishing keyword in path', 15))

    # Signal 12: High numeric character ratio (mild = 5 pts)
    if feat_dict.get('percentage_numeric_chars', 0) > 0.15:
        score += 5
        signals.append(('High numeric ratio', 5))

    max_score = 180
    return score, max_score, signals


# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.markdown("## Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["URL Checker", "Log Anomaly Viewer", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Graph-Based Adaptive Phishing Detection**")
st.sidebar.markdown("Group 19-10 | SOA University ITER")
st.sidebar.markdown("Supervised by Dr. Mohammad Maksood Akhtar")
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset URLs: {len(url_lookup):,}  |  "
    f"0.0=Phishing  |  1.0=Legitimate"
)


# ================================================================
# PAGE 1 -- URL CHECKER
# ================================================================
if page == "URL Checker":

    st.title("Real-Time URL Phishing Checker")
    st.markdown(
        "Detection uses a **4-layer approach** based on best practices "
        "from published phishing detection research:\n\n"
        "**1. Dataset Lookup** — exact match in 100k+ training URLs\n\n"
        "**2. Whitelist** — known safe domains always pass\n\n"
        "**3. Score-Based** — weighted suspicious signal scoring "
        "(IP address, brand impersonation, suspicious TLD, no HTTPS etc.)\n\n"
        "**4. ML Model** — Random Forest on 16 URL features as fallback"
    )
    st.info("**0.0 = Phishing  |  1.0 = Legitimate**")
    st.markdown("---")

    url_input = st.text_input(
        "Enter URL to analyse:",
        placeholder="https://example.com/login",
        help="Include full URL with http:// or https://"
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        analyse = st.button("Analyse", type="primary")

    if analyse and url_input.strip():
        with st.spinner("Analysing..."):

            url_clean = url_input.strip()
            feat_dict = extract_url_features(url_clean)

            if feat_dict is None:
                st.error("Could not parse the URL. Check format and try again.")
            else:
                decision_source = 'ML Model'
                prediction      = None
                proba           = None
                score_info      = None

                # ── Layer 1: Dataset Lookup ───────────────────────
                if url_clean in url_lookup:
                    label      = float(url_lookup[url_clean])
                    prediction = label
                    proba      = np.array([0.99, 0.01]) if label == 0.0 \
                                 else np.array([0.01, 0.99])
                    decision_source = 'Dataset Lookup'

                else:
                    parsed = urlparse(url_clean)
                    domain = parsed.netloc.lower()

                    # ── Layer 2: Whitelist ────────────────────────
                    is_known_legit = any(
                        domain == d or domain.endswith('.' + d)
                        for d in KNOWN_LEGIT
                    )

                    if is_known_legit:
                        prediction      = 1.0
                        proba           = np.array([0.02, 0.98])
                        decision_source = 'Whitelist'

                    else:
                        # ── Layer 3: Score-Based Detection ────────
                        score, max_score, signals = compute_phishing_score(
                            url_clean, feat_dict
                        )
                        score_info = (score, max_score, signals)

                        # Threshold = 25 points out of 180
                        # Any single strong signal (IP, @, brand) triggers this
                        if score >= 25:
                            confidence  = min(0.50 + (score / max_score) * 0.49, 0.99)
                            prediction  = 0.0
                            proba       = np.array([confidence, 1 - confidence])
                            decision_source = f'Score-Based ({score}/{max_score} pts)'

                        else:
                            # ── Layer 4: ML Model ─────────────────
                            feat_df     = align_features(feat_dict)
                            feat_scaled = scaler.transform(feat_df)
                            prediction  = model.predict(feat_scaled)[0]
                            proba       = model.predict_proba(feat_scaled)[0]
                            decision_source = 'ML Model'

                # Final confidence values
                # 0.0=Phishing -> proba[0], 1.0=Legitimate -> proba[1]
                phish_conf = proba[0] * 100
                legit_conf = proba[1] * 100

                st.markdown("---")

                if prediction == 0.0:
                    st.error(
                        f"## PHISHING DETECTED\n\n"
                        f"Confidence: **{phish_conf:.1f}%**"
                    )
                else:
                    st.success(
                        f"## LEGITIMATE URL\n\n"
                        f"Confidence: **{legit_conf:.1f}%**"
                    )

                st.markdown("### Detection Details")
                c1, c2, c3 = st.columns(3)
                c1.metric("Phishing Probability",  f"{phish_conf:.1f}%")
                c2.metric("Legitimate Probability", f"{legit_conf:.1f}%")
                c3.metric("Decision Layer",          decision_source)

                # Show score signals if score-based decision was made
                if score_info:
                    score, max_score, signals = score_info
                    st.markdown(f"### Suspicious Signal Score: {score}/{max_score}")
                    if signals:
                        signal_df = pd.DataFrame(
                            signals, columns=['Signal', 'Points']
                        )
                        st.dataframe(signal_df, use_container_width=True,
                                     hide_index=True)
                    else:
                        st.info("No suspicious signals detected — ML Model used.")

                st.markdown("### Feature Breakdown")
                feat_display = pd.DataFrame(
                    list(feat_dict.items()),
                    columns=['Feature', 'Value']
                )

                def highlight_suspicious(row):
                    flags = {
                        'has_ip_address'           : lambda v: v == 1,
                        'https_flag'               : lambda v: v == 0,
                        'url_length'               : lambda v: v > 75,
                        'subdomain_count'          : lambda v: v >= 3,
                        'has_hyphen_in_domain'     : lambda v: v == 1,
                        'suspicious_file_extension': lambda v: v == 1,
                        'number_of_digits'         : lambda v: v > 10,
                        'tld_popularity'           : lambda v: v == 0,
                        'dot_count'                : lambda v: v >= 5,
                    }
                    if row['Feature'] in flags:
                        if flags[row['Feature']](row['Value']):
                            return ['background-color: #FFE0E0',
                                    'background-color: #FFE0E0']
                    return ['', '']

                styled = feat_display.style.apply(highlight_suspicious, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.caption("Red = suspicious feature value.")

    elif analyse and not url_input.strip():
        st.warning("Please enter a URL before clicking Analyse.")

    st.markdown("---")
    st.markdown("### Sample URLs to Test")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Phishing (should be PHISHING):**")
        st.code("http://192.168.1.1/login/verify.php")
        st.code("http://paypal-secure.verify-account.xyz/confirm")
        st.code("http://bank-login.suspicious-domain.tk/@secure")
        st.code("http://amazon-account.update-verify.com/signin")
        st.code("http://google-security-alert.xyz/verify")
    with c2:
        st.markdown("**Legitimate (should be LEGITIMATE):**")
        st.code("https://www.google.com")
        st.code("https://github.com/login")
        st.code("https://www.microsoft.com/en-us/")
        st.code("https://stackoverflow.com/questions")
        st.code("https://www.paypal.com/signin")


# ================================================================
# PAGE 2 -- LOG ANOMALY VIEWER
# ================================================================
elif page == "Log Anomaly Viewer":

    st.title("Log Anomaly Detection Results")
    st.markdown(
        "Anomaly detection results from both synthetic logs "
        "and real CICIDS2017 network attack data."
    )
    st.markdown("---")

    tab1, tab2 = st.tabs([
        "Synthetic Logs (Phase 4)",
        "CICIDS2017 Real Attacks (Phase 4b)"
    ])

    # ── Tab 1: Synthetic Logs ─────────────────────────────────────
    with tab1:
        st.markdown("### Synthetic Log Anomaly Detection")
        st.markdown(
            "Results from Isolation Forest on synthetic server "
            "access logs with 3 injected attack IPs."
        )

        if not os.path.exists(LOG_PATH):
            st.warning(
                "Log features file not found. "
                "Run Phase 4 notebook first."
            )
        else:
            log_df = pd.read_csv(LOG_PATH)
            total   = len(log_df)
            anomaly = (log_df['anomaly_label'] == 'ANOMALY').sum()
            normal  = total - anomaly

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total IPs",   f"{total}")
            c2.metric("Anomalous",   f"{anomaly}",
                      delta=f"{anomaly/total*100:.1f}%",
                      delta_color="inverse")
            c3.metric("Normal",      f"{normal}")
            c4.metric("Algorithm",   "Isolation Forest")

            st.markdown("---")
            filter_opt = st.radio(
                "Show:", ["All IPs", "Anomalous Only", "Normal Only"],
                horizontal=True, key="syn_filter"
            )
            if filter_opt == "Anomalous Only":
                display_df = log_df[log_df['anomaly_label'] == 'ANOMALY']
            elif filter_opt == "Normal Only":
                display_df = log_df[log_df['anomaly_label'] == 'Normal']
            else:
                display_df = log_df

            display_df = display_df.sort_values(
                'anomaly_score_norm', ascending=False
            ).reset_index(drop=True)

            def colour_anomaly(row):
                if row['anomaly_label'] == 'ANOMALY':
                    return ['background-color: #FFE8E8'] * len(row)
                return [''] * len(row)

            cols_to_show = [
                'ip', 'request_count', 'unique_urls', 'error_rate',
                'avg_size', 'post_ratio', 'unique_url_ratio',
                'anomaly_score_norm', 'anomaly_label'
            ]
            available = [c for c in cols_to_show
                         if c in display_df.columns]

            styled_log = display_df[available].style.apply(
                colour_anomaly, axis=1
            ).format({
                'error_rate'        : '{:.2f}',
                'post_ratio'        : '{:.2f}',
                'unique_url_ratio'  : '{:.2f}',
                'avg_size'          : '{:.0f}',
                'anomaly_score_norm': '{:.4f}',
            })
            st.dataframe(styled_log, use_container_width=True,
                         hide_index=True)
            st.caption(
                "Red = anomalous IP.  "
                "Score range: 0 (normal) to 1 (most anomalous)."
            )

            st.markdown("---")
            st.markdown("### Known Attack IP Profiles")
            known = ['10.0.0.99', '185.220.101.45', '172.16.5.200']
            attack_names = {
                '10.0.0.99'      : 'Brute-Force Login',
                '185.220.101.45' : 'Directory Scanner',
                '172.16.5.200'   : 'Bot Traffic'
            }
            for ip in known:
                rows = log_df[log_df['ip'] == ip]
                if len(rows) > 0:
                    row   = rows.iloc[0]
                    label = row.get('anomaly_label', 'Unknown')
                    with st.expander(
                        f"{ip} — {attack_names.get(ip)} -> {label}"
                    ):
                        detail_cols = [c for c in available
                                       if c != 'ip']
                        st.dataframe(
                            pd.DataFrame([row[detail_cols].to_dict()]),
                            use_container_width=True, hide_index=True
                        )

    # ── Tab 2: CICIDS2017 ─────────────────────────────────────────
    with tab2:
        st.markdown("### CICIDS2017 Real Attack Detection")
        st.markdown(
            "Validation on real network attack data from the "
            "Canadian Institute for Cybersecurity (2017). "
            "Contains DoS Hulk, DoS slowloris, and DoS Slowhttptest attacks."
        )

        CICIDS_PATH   = os.path.join(
            BASE, 'data', 'processed', 'cicids_top_anomalies.csv'
        )
        CICIDS_REPORT = os.path.join(
            BASE, 'reports', 'cicids_report.json'
        )
        CICIDS_IMG    = os.path.join(
            BASE, 'reports', 'cicids_anomaly_results.png'
        )

        if not os.path.exists(CICIDS_REPORT):
            st.warning(
                "CICIDS results not found. "
                "Run notebook 04b_cicids_analysis.ipynb first."
            )
        else:
            with open(CICIDS_REPORT) as f:
                cicids_data = json.load(f)

            r = cicids_data['results']

            # Summary metrics
            st.markdown("### Detection Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Flows",
                      f"{cicids_data['total_flows']:,}")
            c2.metric("Attack Flows",
                      f"{cicids_data['attack_flows']:,}")
            c3.metric("Detection Rate",
                      f"{r['detection_rate']}%")
            c4.metric("F1 Score",
                      f"{r['f1_score']}")

            st.markdown("---")

            # Method comparison
            st.markdown("### Method Comparison")
            comparison_df = pd.DataFrame([
                {
                    'Method'        : 'Isolation Forest (unsupervised)',
                    'Detection Rate': '44.9%',
                    'False Alarm'   : '30.6%',
                    'F1 Score'      : '0.4491',
                    'Labels Needed' : 'No'
                },
                {
                    'Method'        : 'Random Forest (supervised)',
                    'Detection Rate': '99.6%',
                    'False Alarm'   : '1.8%',
                    'F1 Score'      : '0.9815',
                    'Labels Needed' : 'Yes'
                }
            ])
            st.dataframe(comparison_df, use_container_width=True,
                         hide_index=True)
            st.caption(
                "Supervised Random Forest achieves significantly higher "
                "detection including slow-rate DoS attacks that evade "
                "unsupervised Isolation Forest."
            )

            st.markdown("---")

            # Per attack type
            st.markdown("### Per Attack Type Detection")
            attack_df = pd.DataFrame([
                {
                    'Attack Type'    : 'DoS Hulk (high volume)',
                    'Total Flows'    : 24364,
                    'IF Detected'    : '47.5%',
                    'RF Detected'    : '99.7%',
                    'Evasion Risk'   : 'Low'
                },
                {
                    'Attack Type'    : 'DoS slowloris (slow rate)',
                    'Total Flows'    : 5796,
                    'IF Detected'    : '34.4%',
                    'RF Detected'    : '99.8%',
                    'Evasion Risk'   : 'High'
                },
                {
                    'Attack Type'    : 'DoS Slowhttptest (slow rate)',
                    'Total Flows'    : 5499,
                    'IF Detected'    : '44.8%',
                    'RF Detected'    : '99.9%',
                    'Evasion Risk'   : 'High'
                }
            ])

            def highlight_evasion(row):
                if row['Evasion Risk'] == 'High':
                    return ['background-color: #FFE8E8'] * len(row)
                return ['background-color: #E8FFE8'] * len(row)

            st.dataframe(
                attack_df.style.apply(highlight_evasion, axis=1),
                use_container_width=True, hide_index=True
            )
            st.caption(
                "IF = Isolation Forest  |  RF = Random Forest  |  "
                "Red = high evasion risk attacks"
            )

            st.markdown("---")

            # Results chart
            if os.path.exists(CICIDS_IMG):
                st.markdown("### Detection Results Chart")
                st.image(CICIDS_IMG, use_column_width=True)

            # Finding
            st.markdown("---")
            st.info(
                "**Key Finding:** Slow-rate DoS attacks (slowloris, "
                "Slowhttptest) deliberately mimic normal traffic to evade "
                "anomaly detection. Unsupervised Isolation Forest detects "
                "only 34-45% of these attacks. Supervised Random Forest "
                "with labeled training data achieves 99%+ detection. "
                "This demonstrates why our adaptive retraining module "
                "(Phase 6) is critical — continuous retraining on new "
                "labeled attack data maintains high detection accuracy."
            )
# ================================================================
# PAGE 3 -- MODEL PERFORMANCE
# ================================================================
elif page == "Model Performance":

    st.title("Model Performance and Retraining History")
    st.info("**0.0 = Phishing  |  1.0 = Legitimate**")
    st.markdown("---")

    st.markdown("### Classification Performance")
    tab1, tab2, tab3 = st.tabs(
        ["ROC Curves", "Confusion Matrices", "Feature Importance"]
    )

    with tab1:
        if os.path.exists(ROC_IMG):
            st.image(ROC_IMG, use_column_width=True)
            st.caption("AUC closer to 1.0 = better. Positive class = Phishing (0.0).")
        else:
            st.warning("Run Phase 5 notebook.")

    with tab2:
        if os.path.exists(CM_IMG):
            st.image(CM_IMG, use_column_width=True)
            st.caption("Row 0 = Phishing (0.0)  |  Row 1 = Legitimate (1.0).")
        else:
            st.warning("Run Phase 5 notebook.")

    with tab3:
        if os.path.exists(FI_IMG):
            st.image(FI_IMG, use_column_width=True)
            st.caption("Orange = graph-derived  |  Blue = URL structural.")
        else:
            st.warning("Run Phase 5 notebook.")

    st.markdown("---")
    st.markdown("### Adaptive Retraining History")

    if os.path.exists(RH_IMG):
        st.image(RH_IMG, use_column_width=True)

    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG) as f:
            retrain_log = json.load(f)

        total_checks  = len(retrain_log)
        retrain_count = sum(1 for e in retrain_log if e['retrained'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Checks",    total_checks)
        c2.metric("Retraining Events", retrain_count)
        c3.metric("No Action",       total_checks - retrain_count)
        c4.metric("F1 Threshold",    "0.90 macro")

        log_df_display = pd.DataFrame(retrain_log)

        def colour_retrained(row):
            if row.get('retrained', False):
                return ['background-color: #FFE8E8'] * len(row)
            return ['background-color: #E8FFE8'] * len(row)

        st.dataframe(
            log_df_display.style.apply(colour_retrained, axis=1),
            use_container_width=True, hide_index=True
        )
        st.caption("Green = healthy.  Red = retraining triggered.")
    else:
        st.warning("Run Phase 6 notebook.")

    st.markdown("---")
    st.markdown("### Phase Reports")
    for name, rel_path in {
        'Phase 5': 'reports/phase5_report.json',
        'Phase 6': 'reports/phase6_report.json',
        'Phase 4': 'reports/phase4_report.json',
    }.items():
        full_path = os.path.join(BASE, rel_path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                data = json.load(f)
            with st.expander(f"{name} Report"):
                st.json(data)