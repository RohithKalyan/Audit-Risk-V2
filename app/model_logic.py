# === CORRECTED model_logic.py - 65 Columns with August 25 Integration ===

import pandas as pd
import numpy as np
import os
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union
warnings.filterwarnings("ignore")

# === BERT RISK EXPLAINER CLASS (Internal Use Only) ===
class BERTRiskExplainer:
    def __init__(self):
        self.business_risk_patterns = {
            'sunday_payment_processing': {
                'trigger': self._check_sunday_payment_processing,
                'explanation': "Sunday payment processing bypassing standard authorization controls",
            },
            'vague_account_classification': {
                'trigger': self._check_vague_account_classification,
                'explanation': "Vague account classifications lacking transaction specificity",
            },
            'high_value_escrow_processing': {
                'trigger': self._check_high_value_escrow_processing,
                'explanation': "High-value escrow processing requiring enhanced fiduciary oversight",
            },
            'system_integration_processing': {
                'trigger': self._check_system_integration_processing,
                'explanation': "System integration processing with data integrity vulnerabilities",
            },
            'manual_ecommerce_operations': {
                'trigger': self._check_manual_ecommerce_operations,
                'explanation': "Manual e-commerce operations bypassing automated controls",
            },
            'cod_settlement_verification': {
                'trigger': self._check_cod_settlement_verification,
                'explanation': "COD settlement with logistics coordination timing differences",
            },
            'payment_gateway_reconciliation': {
                'trigger': self._check_payment_gateway_reconciliation,
                'explanation': "Payment gateway requiring multi-party reconciliation processes",
            },
            'revenue_recognition_timing': {
                'trigger': self._check_revenue_recognition_timing,
                'explanation': "Revenue recognition timing requiring compliance assessment",
            },
            'adjustment_entry_documentation': {
                'trigger': self._check_adjustment_entry_documentation,
                'explanation': "Manual adjustments deviating from standard processing workflows",
            }
        }
    
    def _check_sunday_payment_processing(self, row_data: Dict, text: str) -> bool:
        try:
            day = str(row_data.get('Day', '')).strip()
            sunday_days = ['Sun', 'Sunday']
            payment_terms = ['wallet', 'hadoop', 'payment', 'cashfree']
            return (day in sunday_days and any(term in text.lower() for term in payment_terms))
        except: return False
    
    def _check_vague_account_classification(self, row_data: Dict, text: str) -> bool:
        try:
            account_name = str(row_data.get('Account Name', '')).lower()
            adjustment_terms = ['adjustment', 'settlement', 'liability']
            return ('other' in account_name and any(term in text.lower() for term in adjustment_terms))
        except: return False
    
    def _check_high_value_escrow_processing(self, row_data: Dict, text: str) -> bool:
        try:
            escrow_terms = ['escrow', 'wallet', 'liability']
            account_name = str(row_data.get('Account Name', '')).lower()
            amount = self._safe_float_conversion(row_data.get('Net Amount', 0))
            return (any(term in text.lower() for term in escrow_terms) and
                    'escrow' in account_name and abs(amount) > 500000)
        except: return False
    
    def _check_system_integration_processing(self, row_data: Dict, text: str) -> bool:
        try:
            system_terms = ['hadoop', 'system', 'automated', 'verified', 'matched']
            processing_terms = ['processing', 'settlement', 'reconciliation']
            return (any(term in text.lower() for term in system_terms) and
                    any(term in text.lower() for term in processing_terms))
        except: return False
    
    def _check_manual_ecommerce_operations(self, row_data: Dict, text: str) -> bool:
        try:
            manual_terms = ['spreadsheet', 'manual']
            ecommerce_terms = ['gmv', 'seller', 'rebate', 'voucher']
            amount = self._safe_float_conversion(row_data.get('Net Amount', 0))
            return (any(term in text.lower() for term in manual_terms) and
                    any(term in text.lower() for term in ecommerce_terms) and
                    abs(amount) > 1000000)
        except: return False
    
    def _check_cod_settlement_verification(self, row_data: Dict, text: str) -> bool:
        try:
            cod_terms = ['cod', 'delhivery', 'delivery']
            settlement_terms = ['settlement', 'collection', 'payment']
            return (any(term in text.lower() for term in cod_terms) and
                    any(term in text.lower() for term in settlement_terms))
        except: return False
    
    def _check_payment_gateway_reconciliation(self, row_data: Dict, text: str) -> bool:
        try:
            gateway_terms = ['cashfree', 'gateway', 'payment']
            process_terms = ['settlement', 'reconciliation', 'processing']
            return (any(term in text.lower() for term in gateway_terms) and
                    any(term in text.lower() for term in process_terms))
        except: return False
    
    def _check_revenue_recognition_timing(self, row_data: Dict, text: str) -> bool:
        try:
            pl_bs = str(row_data.get('PL/ BS', '')).upper().strip()
            revenue_terms = ['revenue', 'income', 'sales']
            amount = self._safe_float_conversion(row_data.get('Net Amount', 0))
            return (pl_bs == 'PL' and
                    any(term in text.lower() for term in revenue_terms) and
                    abs(amount) > 500000)
        except: return False
    
    def _check_adjustment_entry_documentation(self, row_data: Dict, text: str) -> bool:
        try:
            adjustment_terms = ['adjustment', 'correction', 'manual']
            process_terms = ['settlement', 'reconciliation', 'variance']
            return (any(term in text.lower() for term in adjustment_terms) and
                    any(term in text.lower() for term in process_terms))
        except: return False
    
    def _safe_float_conversion(self, value: Union[str, int, float], default: float = 0.0) -> float:
        try:
            if pd.isna(value) or value is None: return default
            return float(value)
        except: return default
    
    def explain_bert_risk(self, transaction_data: Dict, bert_impact: float) -> Optional[str]:
        try:
            if pd.isna(bert_impact) or bert_impact < 0.05: return None
            
            text = str(transaction_data.get('Combined_Text', '')).lower().strip()
            if not text or text == 'nan': return None
            
            # Check business risk patterns first
            for pattern_name, pattern_config in self.business_risk_patterns.items():
                try:
                    if pattern_config['trigger'](transaction_data, text):
                        return pattern_config['explanation']
                except: continue
            
            # Fallback explanations based on text patterns
            if any(term in text for term in ['other', 'miscellaneous', 'various', 'general']):
                return "Vague transaction descriptions lacking specific business purpose"
            elif any(term in text for term in ['spreadsheet', 'manual', 'excel']):
                return "Manual processing bypassing automated control frameworks"
            elif any(term in text for term in ['urgent', 'emergency', 'immediate']):
                return "Urgency indicators suggesting potential workflow bypass"
            else:
                return "Text-based risk patterns requiring enhanced verification"
                
        except Exception as e:
            return None

def calculate_bert_impact(shap_values: np.ndarray, feature_names: List[str]) -> float:
    try:
        bert_features = [i for i, name in enumerate(feature_names) if name.startswith('text_emb_')]
        bert_impact = sum(shap_values[i] for i in bert_features if shap_values[i] > 0)
        return bert_impact
    except: return 0.0

def check_bert_in_top3(shap_values: np.ndarray, feature_names: List[str]) -> Tuple[bool, float]:
    try:
        feature_impacts = [(feature, shap_val) for feature, shap_val in zip(feature_names, shap_values) if shap_val > 0]
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        top_3_features = feature_impacts[:3]
        bert_in_top3 = any(feature.startswith('text_emb_') for feature, shap_val in top_3_features)
        bert_impact = calculate_bert_impact(shap_values, feature_names)
        return bert_in_top3, bert_impact
    except: return False, 0.0

def get_auditor_friendly_explanation(feature, value, shap_impact):
    try:
        if feature == "Day":
            try:
                day_num = int(value)
                if day_num == 6:  # Sunday
                    return "Transaction processed on Sunday when standard business operations are typically not active"
                else: return None
            except: return None
        
        elif feature == "Account Name":
            value_str = str(value).lower()
            if "other debtors" in value_str:
                return "Vague account classification lacking proper transaction specificity"
            elif "other professional fees" in value_str:
                return "General professional fee account lacking vendor specificity"
            elif "legal fee" in value_str:
                return "Legal fee account susceptible to inappropriate payments"
            elif "receivables from cod" in value_str:
                return "Cash-on-delivery receivables requiring enhanced verification"
            elif "cash in bank" in value_str:
                return "Cash account requiring verification of bank reconciliations"
            elif "receivables from payment gateway" in value_str:
                return "Payment gateway receivables requiring verification of settlement timing"
            else: return None
        
        elif feature == "Nature in balance sheet":
            value_str = str(value).lower()
            if "logistic" in value_str and "packing" in value_str:
                return "Logistics expense category prone to cost inflation"
            elif "legal" in value_str and "professional" in value_str:
                return "Professional services expense susceptible to manipulation"
            elif "provision" in value_str:
                return "Provision account lacking detailed substantiation"
            else: return None
        
        elif feature in ["Net Amount", "Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM"]:
            try:
                amount = float(value)
                if amount >= 10000000000:  # 1000 Cr
                    return "Exceptionally high transaction value representing significant financial exposure"
                elif amount >= 5000000000:  # 500 Cr
                    return "Very high transaction value exceeding typical business thresholds"
                elif amount >= 1000000000:  # 100 Cr
                    return "High transaction value exceeding standard materiality thresholds"
                elif amount >= 500000000:  # 50 Cr
                    return "Material transaction amount warranting enhanced scrutiny"
                else: return None
            except: return "Transaction amount requiring verification due to data quality issues"
        
        elif feature == "Batch Name":
            value_str = str(value).lower()
            if "spreadsheet" in value_str:
                return "Bulk spreadsheet processing bypassing individual transaction controls"
            else: return None
        
        elif feature == "Document Type":
            if str(value) == "Manual":
                return "Manual entry increasing error risk and bypassing automated validation"
            elif str(value) == "Spreadsheet":
                return "Spreadsheet-based entry bypassing automated controls"
            else: return None
        
        elif "Weekday" in feature:
            try:
                day_num = int(value)
                if day_num == 6:  # Sunday only
                    return "Transaction processed on Sunday when standard business operations are typically not active"
                else: return None
            except: return None
        
        else: return None
    
    except Exception as e:
        return None

def run_full_pipeline(file_path: str) -> pd.DataFrame:
    logging.debug("Starting run_full_pipeline")
    # Lazy load heavy packages here
    import shap
    from catboost import CatBoostClassifier
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from sklearn.cluster import KMeans
    import requests
    from io import BytesIO

    # === Load Models ===
    try:
        print("Loading CatBoost model...")
        model = CatBoostClassifier()
        model.load_model("models/catboost_v2_model.cbm")
        print("CatBoost model loaded")
    except Exception as e:
        print("Failed to load CatBoost model:", str(e))
        raise

    try:
        print("Downloading SentenceTransformer...")
        model_bert = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceTransformer loaded")
    except Exception as e:
        print("Failed to load BERT model:", str(e))
        raise

    # === Load and Clean Data ===
    test_df = pd.read_csv(file_path, encoding='ISO-8859-1')
    test_df.columns = test_df.columns.str.strip()

    # Clean numeric columns
    comma_cols = ["Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM", "Net Amount"]
    for col in comma_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str).str.replace(",", "").replace("nan", np.nan).astype(float)

    # Combine text fields
    text_fields = ["Line Desc", "Source Desc", "Batch Name"]
    test_df[text_fields] = test_df[text_fields].fillna("")
    test_df["Combined_Text"] = test_df["Line Desc"] + " | " + test_df["Source Desc"] + " | " + test_df["Batch Name"]

    # === BERT Embeddings and Clustering ===
    embeddings = model_bert.encode(test_df["Combined_Text"].tolist(), show_progress_bar=False)
    embedding_df = pd.DataFrame(embeddings, columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])])
    test_df = pd.concat([test_df.reset_index(drop=True), embedding_df], axis=1)

    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced = umap_model.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=10, random_state=42)
    test_df["Narration_Cluster"] = kmeans.fit_predict(reduced)

    cluster_summary = (
        test_df.groupby("Narration_Cluster")["Combined_Text"]
        .apply(lambda x: "; ".join(x.head(3)))
        .reset_index(name="Narration_Cluster_Label")
    )
    test_df = test_df.merge(cluster_summary, on="Narration_Cluster", how="left")

    # === Date Features ===
    date_cols = ["Accounting Date", "Invoice Date", "Posted Date"]
    for col in date_cols:
        test_df[col] = pd.to_datetime(test_df[col], errors="coerce")

    test_df["Accounting_Month"] = test_df["Accounting Date"].dt.month
    test_df["Accounting_Weekday"] = test_df["Accounting Date"].dt.weekday
    test_df["Invoice_Month"] = test_df["Invoice Date"].dt.month
    test_df["Invoice_Weekday"] = test_df["Invoice Date"].dt.weekday
    test_df["Posted_Month"] = test_df["Posted Date"].dt.month
    test_df["Posted_Weekday"] = test_df["Posted Date"].dt.weekday

    # === Feature Preparation ===
    exclude_cols = ["S. No", "Combined_Text", "Accounting Date", "Invoice Date", "Posted Date"]
    model_feature_names = model.feature_names_
    feature_cols = [col for col in test_df.columns if col in model_feature_names and col not in exclude_cols and not col.startswith("Unnamed")]

    for col in feature_cols:
        if test_df[col].dtype == object or test_df[col].isnull().any():
            test_df[col] = test_df[col].astype(str).fillna("Missing")

    X_final = test_df[feature_cols].copy()

    # === Model Predictions ===
    test_df["Model_Score"] = model.predict_proba(X_final)[:, 1]
    test_df["Final_Score"] = test_df["Model_Score"].round(3)

    # === SHAP Analysis ===
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_final)

    # === Initialize BERT Explainer ===
    bert_explainer = BERTRiskExplainer()

    # === Control Points Setup ===
    cp_score_dict = {
        "CP_01": 83, "CP_02": 86, "CP_03": 78, "CP_04": 81, "CP_07": 84, "CP_08": 80,
        "CP_09": 76, "CP_15": 88, "CP_16": 73, "CP_17": 75, "CP_19": 60,
        "CP_21": 69, "CP_22": 66, "CP_23": 87, "CP_24": 78, "CP_26": 0,
        "CP_30": 72, "CP_32": 72
    }
    valid_cps = list(cp_score_dict.keys())

    pl_net_total = test_df[test_df["PL/ BS"] == "PL"]["Net Amount"].abs().sum()
    pl_net_threshold = 0.10 * pl_net_total
    total_net = test_df["Net Amount"].abs().sum()

    # === Control Point Functions ===
    def cp_01(row):
        keywords = ['fraud','bribe','kickback','suspicious','fake','dummy','gift','prize','token','reward','favour']
        text = f"{str(row.get('Line Desc', '')).lower()} {str(row.get('Source Desc', '')).lower()}"
        return int(any(k in text for k in keywords))

    def cp_02(row):
        return int(row.get("PL/ BS") == "PL" and abs(row.get("Net Amount", 0)) > pl_net_threshold)

    def cp_03_flags(df):
        a = df.duplicated(subset=["Accounting Date", "Line Desc", "Source Desc", "Source Name"], keep=False)
        b = df.duplicated(subset=["Accounting Date", "Account Code", "Net Amount"], keep=False)
        c = df.duplicated(subset=["Document Number"], keep=False) & ~df.duplicated(subset=["Accounting Date", "Document Number"], keep=False)
        d = df.duplicated(subset=["Accounting Date", "Line Desc", "Account Code"], keep=False)
        return ((a | b | c | d).astype(int))

    def cp_04(row): return cp_02(row)

    def cp_07_flags(df): return (df.groupby("Document Number")["Net Amount"].transform("sum").round(2) != 0).astype(int)

    def cp_08(row):
        text = f"{row.get('Account Name', '')} {row.get('Line Desc', '')} {row.get('Source Desc', '')}".lower()
        return int("cash in hand" in text)

    def cp_09_flags(df):
        result = pd.Series(0, index=df.index)
        for doc_id, group in df.groupby("Document Number"):
            accs = group["Account Name"].dropna().str.lower().tolist()
            if any("cash" in a for a in accs) and any("bad debt" in a for a in accs):
                result[group.index] = 1
        return result

    def cp_15_flags(df):
        grp_sum = df.groupby(["Account Code", "Accounting Date"])[["Entered Dr SUM", "Entered Cr SUM"]].sum().sum(axis=1)
        keys = grp_sum[grp_sum > 0.03 * total_net].index
        return df.set_index(["Account Code", "Accounting Date"]).index.isin(keys).astype(int)

    def cp_16_flags(df):
        if "Currency" not in df.columns:
            df["Currency"] = "INR"
        docs = df.groupby("Document Number")["Currency"].nunique()
        flagged = docs[docs > 1].index
        return df["Document Number"].isin(flagged).astype(int)

    def cp_17_flags(df):
        sums = df[df["PL/ BS"] == "PL"].groupby("Source Name")["Net Amount"].sum().abs()
        risky = sums[sums > 0.03 * pl_net_total].index
        return df["Source Name"].isin(risky).astype(int)

    def cp_19(row):
        try: return int(pd.to_datetime(row["Accounting Date"]).weekday() == 6)
        except: return 0

    def cp_21(row):
        try:
            date = pd.to_datetime(row.get("Accounting Date"))
            return int(date == (date + pd.offsets.MonthEnd(0)))
        except: return 0

    def cp_22(row):
        try:
            date = pd.to_datetime(row.get("Accounting Date"))
            return int(date.day == 1)
        except: return 0

    def cp_23(row):
        text = f"{row.get('Line Desc', '')} {row.get('Account Name', '')}".lower()
        return int(any(t in text for t in ['derivative', 'spv', 'structured', 'note', 'swap']))

    def cp_24(row):
        try:
            last = str(int(abs(row.get("Net Amount", 0))))[-3:]
            seqs = {'123','234','345','456','567','678','789','890','321','432','543','654','765','876','987','098'}
            repeats = {str(i)*3 for i in range(10)} | {'000'}
            return int(last in seqs or last in repeats and last != '901')
        except: return 0

    def cp_26_flags(df):
        try:
            doc_ids = sorted(df["Document Number"].dropna().astype(int).unique())
            missing = {doc_ids[i]+1 for i in range(len(doc_ids)-1) if doc_ids[i+1] - doc_ids[i] > 1}
            flagged = set()
            for miss in missing:
                flagged.update([miss-1, miss+1])
            return df["Document Number"].astype(int).isin(flagged).astype(int)
        except: return pd.Series(0, index=df.index)

    def cp_30(row):
        text = f"{row.get('Line Desc', '')} {row.get('Account Name', '')}".lower()
        return int(any(t in text for t in ['derivative','option','swap','future','structured']))

    def cp_32(row): return int(row.get("Net Amount", 0) == 0)

    # === Apply All Control Points ===
    test_df["CP_01"] = test_df.apply(cp_01, axis=1)
    test_df["CP_02"] = test_df.apply(cp_02, axis=1)
    test_df["CP_03"] = cp_03_flags(test_df)
    test_df["CP_04"] = test_df.apply(cp_04, axis=1)
    test_df["CP_07"] = cp_07_flags(test_df)
    test_df["CP_08"] = test_df.apply(cp_08, axis=1)
    test_df["CP_09"] = cp_09_flags(test_df)
    test_df["CP_15"] = cp_15_flags(test_df)
    
    # Ensure Currency column is created before CP_16
    if "Currency" not in test_df.columns:
        test_df["Currency"] = "INR"
    test_df["CP_16"] = cp_16_flags(test_df)
    
    test_df["CP_17"] = cp_17_flags(test_df)
    test_df["CP_19"] = test_df.apply(cp_19, axis=1)
    test_df["CP_21"] = test_df.apply(cp_21, axis=1)
    test_df["CP_22"] = test_df.apply(cp_22, axis=1)
    test_df["CP_23"] = test_df.apply(cp_23, axis=1)
    test_df["CP_24"] = test_df.apply(cp_24, axis=1)
    test_df["CP_26"] = cp_26_flags(test_df)
    test_df["CP_30"] = test_df.apply(cp_30, axis=1)
    test_df["CP_32"] = test_df.apply(cp_32, axis=1)

    def compute_cp_score(row):
        triggered = [cp for cp in valid_cps if row.get(cp, 0) == 1]
        if not triggered: return 0.0
        product = 1.0
        for cp in triggered:
            product *= (1 - cp_score_dict[cp] / 100)
        return round(1 - product, 4)

    def list_triggered_cps(row):
        return ", ".join([f"{cp} ({cp_score_dict[cp]})" for cp in valid_cps if row.get(cp, 0) == 1])

    test_df["Triggered_CPs"] = test_df.apply(list_triggered_cps, axis=1)
    test_df["CP_Score"] = test_df.apply(compute_cp_score, axis=1)

    # === Enhanced Risk Classifications (INTERNAL LISTS ONLY) ===
    model_class_list = []
    cp_class_list = []
    final_risk_list = []
    
    for i in range(len(test_df)):
        score = test_df.iloc[i]["Final_Score"]
        cp_score = test_df.iloc[i]["CP_Score"]
        
        model_class = "High" if score >= 0.995 else ("Medium" if score >= 0.5 else "Low")
        cp_class = "High" if cp_score >= 0.95 else ("Medium" if cp_score > 0.8 else "Low")
        final_risk = "High" if model_class == "High" or cp_class == "High" else ("Medium" if model_class == "Medium" or cp_class == "Medium" else "Low")
        
        model_class_list.append(model_class)
        cp_class_list.append(cp_class)
        final_risk_list.append(final_risk)

    # === August 25 Enhanced Explanation Generation ===
    def get_cp_explanation(cp_code, row):
        cp_explanations = {
            "CP_01": "Suspicious Keywords - Transaction contains high-risk terms requiring verification",
            "CP_02": f"High Monetary Value - Amount of Rs{row.get('Net Amount', 0):,.0f} exceeds materiality threshold",
            "CP_03": "Duplicate Patterns - Transaction matches multiple duplicate detection criteria",
            "CP_07": "Document Imbalance - Document entries do not balance to zero",
            "CP_08": "Cash Expenditure - Cash-in-hand transaction bypassing standard payment controls",
            "CP_09": "Cash to Bad Debt - Transaction involves both cash and bad debt accounts",
            "CP_15": "Split Transactions - Account activity exceeds normal volume threshold",
            "CP_16": "Multiple Currencies - Document contains multiple currencies",
            "CP_17": "Vendor Concentration - Source transactions exceed concentration limits",
            "CP_19": "Weekend Processing - Transaction processed when standard approvals typically unavailable",
            "CP_21": "Period-End Timing - Transaction occurs on month-end date",
            "CP_22": "Period-Start Timing - Transaction occurs on first day of month",
            "CP_23": "Complex Structure - Transaction involves derivative or structured instruments",
            "CP_24": "Unusual Amount Pattern - Transaction amount follows rare sequential patterns",
            "CP_26": "Document Gap - Document number is missing from sequence",
            "CP_30": "Complex Instrument - Transaction involves sophisticated financial instruments",
            "CP_32": "Zero Amount - Transaction recorded with zero net amount"
        }
        return cp_explanations.get(cp_code, f"Control Point {cp_code} triggered")
    
    def parse_triggered_cps(triggered_cps_str):
        try:
            if not triggered_cps_str or triggered_cps_str.strip() == "": return []
            cp_codes = []
            for cp_part in triggered_cps_str.split(", "):
                if "CP_" in cp_part:
                    cp_code = cp_part.split(" ")[0]
                    cp_codes.append(cp_code)
            return cp_codes
        except: return []

    # Generate Enhanced Explanations using August 25 Logic
    explanation_summaries = []
    
    for i in range(len(X_final)):
        try:
            row_shap = shap_values[i]
            row = test_df.iloc[i]
            final_risk = final_risk_list[i]
            model_class = model_class_list[i]
            cp_class = cp_class_list[i]
            
            # Only generate enhanced explanations for High Risk transactions
            if final_risk == "High":
                bert_in_top3, bert_impact = check_bert_in_top3(row_shap, feature_cols)
                explanations = []
                
                if model_class == "High" and cp_class != "High":
                    # Model-driven risk
                    feature_impacts = []
                    for j, feature in enumerate(feature_cols):
                        if row_shap[j] > 0 and not feature.startswith('text_emb_'):
                            feature_value = row.get(feature, "N/A")
                            feature_impacts.append((feature, feature_value, row_shap[j]))
                    
                    feature_impacts.sort(key=lambda x: x[2], reverse=True)
                    num_regular = 2 if bert_in_top3 and bert_impact >= 0.05 else 3
                    
                    used_explanations = set()
                    consolidated_amounts = []
                    
                    for feature, value, shap_val in feature_impacts:
                        if feature in ["Net Amount", "Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM"]:
                            consolidated_amounts.append((feature, value, shap_val))
                            continue
                        
                        explanation_text = get_auditor_friendly_explanation(feature, value, shap_val)
                        if explanation_text is not None and explanation_text not in used_explanations:
                            formatted_explanation = f'"{feature}: {value}" - {explanation_text}'
                            explanations.append(formatted_explanation)
                            used_explanations.add(explanation_text)
                            if len(explanations) >= num_regular:
                                break
                    
                    if consolidated_amounts and len(explanations) < num_regular:
                        highest_amount = max(consolidated_amounts, key=lambda x: x[2])
                        feature, value, shap_val = highest_amount
                        explanation_text = get_auditor_friendly_explanation(feature, value, shap_val)
                        if explanation_text is not None:
                            formatted_explanation = f'"Transaction Amount: Rs{float(value):,.0f}" - {explanation_text}'
                            explanations.insert(0, formatted_explanation)
                    
                    if bert_in_top3 and bert_impact >= 0.05:
                        row_dict = row.to_dict()
                        bert_exp = bert_explainer.explain_bert_risk(row_dict, bert_impact)
                        if bert_exp:
                            explanations.append(bert_exp)
                    
                elif model_class != "High" and cp_class == "High":
                    # CP-driven risk
                    triggered_cps = parse_triggered_cps(row.get("Triggered_CPs", ""))
                    num_cp = 2 if bert_in_top3 and bert_impact >= 0.05 else 3
                    
                    for cp_code in triggered_cps[:num_cp]:
                        cp_explanation = get_cp_explanation(cp_code, row)
                        explanations.append(cp_explanation)
                    
                    if bert_in_top3 and bert_impact >= 0.05:
                        row_dict = row.to_dict()
                        bert_exp = bert_explainer.explain_bert_risk(row_dict, bert_impact)
                        if bert_exp:
                            explanations.append(bert_exp)
                    
                elif model_class == "High" and cp_class == "High":
                    # Both model and CP high risk
                    num_model = 1 if bert_in_top3 and bert_impact >= 0.05 else 2
                    num_cp = 1
                    
                    # Model features
                    feature_impacts = []
                    for j, feature in enumerate(feature_cols):
                        if row_shap[j] > 0 and not feature.startswith('text_emb_'):
                            feature_value = row.get(feature, "N/A")
                            feature_impacts.append((feature, feature_value, row_shap[j]))
                    
                    feature_impacts.sort(key=lambda x: x[2], reverse=True)
                    
                    for k, (feature, value, shap_val) in enumerate(feature_impacts[:num_model]):
                        explanation_text = get_auditor_friendly_explanation(feature, value, shap_val)
                        if explanation_text is not None:
                            if feature in ["Net Amount", "Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM"]:
                                formatted_explanation = f'"Transaction Amount: Rs{float(value):,.0f}" - {explanation_text}'
                            else:
                                formatted_explanation = f'"{feature}: {value}" - {explanation_text}'
                            explanations.append(formatted_explanation)
                    
                    # CP features
                    triggered_cps = parse_triggered_cps(row.get("Triggered_CPs", ""))
                    for cp_code in triggered_cps[:num_cp]:
                        cp_explanation = get_cp_explanation(cp_code, row)
                        explanations.append(cp_explanation)
                    
                    if bert_in_top3 and bert_impact >= 0.05:
                        row_dict = row.to_dict()
                        bert_exp = bert_explainer.explain_bert_risk(row_dict, bert_impact)
                        if bert_exp:
                            explanations.append(bert_exp)
                
                # Create enhanced explanation for High Risk
                if explanations:
                    final_explanation = "\n".join(explanations[:3])
                    explanation_summaries.append(final_explanation)
                else:
                    explanation_summaries.append("High risk transaction requiring enhanced review")
            else:
                # For non-High risk transactions, use empty explanation
                explanation_summaries.append("")
                
        except Exception as e:
            explanation_summaries.append("")

    # === Generate Original Feature Group Explanations ===
    amount_features = ["Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM", "Net Amount"]
    date_features = ["Accounting_Month", "Accounting_Weekday", "Invoice_Month", "Invoice_Weekday", "Posted_Month", "Posted_Weekday"]
    account_info_features = ["Account Name", "Nature in balance sheet", "Source Name", "Document Type", "Tax Rate", "Tax Rate Name"]
    other_features = [col for col in model.feature_names_ if col not in amount_features + date_features + account_info_features and not col.startswith("text_emb_")]

    feature_groups = {
        "Amount": amount_features,
        "Date": date_features,
        "Source Info": account_info_features,
        "Batch": other_features,
        "Narration": ["Narration_Cluster_Label"]
    }

    explanation_templates = {
        "Narration": "Narration pattern resembles high-value or structured payouts",
        "Amount": "High {feature} = ₹{value:,.0f}",
        "Date": "Posted on {feature} = {value}",
        "Source Info": "{feature} = '{value}' is missing or looks suspicious",
        "Batch": "Batch reference '{value}' appears frequently in vendor payments"
    }

    top_risky_texts, top_safe_texts = [], []
    for i in range(len(X_final)):
        row_shap = shap_values[i]
        row = test_df.iloc[i]

        impact_by_group = {}
        feature_info = {}
        for group, features in feature_groups.items():
            valid_feats = [f for f in features if f in feature_cols]
            if not valid_feats:
                continue
            group_shap_sum = sum(row_shap[feature_cols.index(f)] for f in valid_feats)
            impact_by_group[group] = group_shap_sum
            top_feat = max(valid_feats, key=lambda f: abs(row_shap[feature_cols.index(f)]))
            value = row.get(top_feat, "N/A")
            feature_info[group] = (top_feat, value)

        sorted_risk = sorted(impact_by_group.items(), key=lambda x: x[1], reverse=True)
        sorted_safe = sorted(impact_by_group.items(), key=lambda x: x[1])

        def render(group, feature, value):
            if group == "Narration":
                return explanation_templates[group]
            elif group in explanation_templates:
                return explanation_templates[group].format(feature=feature, value=value)
            else:
                return f"{group}: {feature} = {value}"

        top_risk = [render(g, *feature_info[g]) for g, _ in sorted_risk[:3]]
        top_safe = [render(g, *feature_info[g]) for g, _ in sorted_safe if g not in [r[0] for r in sorted_risk[:3]][:2]]

        top_risky_texts.append("\n".join(f"- {t}" for t in top_risk))
        top_safe_texts.append("\n".join(f"- {t}" for t in top_safe[:2]))

    # Remove BERT embedding columns before creating final columns
    test_df = test_df.drop(columns=[col for col in test_df.columns if col.startswith("text_emb_")])

    # === Create Explanation Columns (Positions 42-44) ===
    test_df["Top_Risky_Feature_Groups"] = top_risky_texts
    test_df["Top_Safe_Feature_Groups"] = top_safe_texts
    test_df["Explanation_Summary"] = explanation_summaries

    # === Final Column Order (65 columns) ===
    expected_columns = [
        # Original 30 columns
        "S. No", "Entity Name", "Accounting Date", "Approval Type", "Document Type", "Invoice Date", "Day", "Nature",
        "Account Code", "PL/ BS", "Report Group", "Account Name", "Nature in balance sheet", "Document Number", "Je Line Num",
        "Source Number", "Source Name", "Source Voucher Name", "Source Desc", "Line Desc", "Project Code", "Internal Reference", 
        "Posted Date", "Branch", "Batch Name", "Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM", "Net Amount",
        
        # Generated analysis columns
        "Combined_Text", "Narration_Cluster", "Narration_Cluster_Label",
        "Accounting_Month", "Accounting_Weekday", "Invoice_Month", "Invoice_Weekday", "Posted_Month", "Posted_Weekday",
        "Model_Score", "Final_Score",
        
        # Explanation columns (positions 42-44)
        "Top_Risky_Feature_Groups", "Top_Safe_Feature_Groups", "Explanation_Summary",
        
        # Control points columns (18 total)
        "CP_01", "CP_02", "CP_03", "CP_04", "CP_07", "CP_08", "CP_09", "CP_15", "Currency", "CP_16", "CP_17", 
        "CP_19", "CP_21", "CP_22", "CP_23", "CP_24", "CP_26", "CP_30", "CP_32", "Triggered_CPs", "CP_Score"
    ]
    
    # Validate all expected columns exist
    missing_columns = [col for col in expected_columns if col not in test_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Reorder columns to exact specification
    final_df = test_df[expected_columns].copy()
    
    # Final validation: ensure exactly 65 columns
    if len(final_df.columns) != 65:
        raise ValueError(f"Expected 65 columns, got {len(final_df.columns)}")
    
    logging.debug("run_full_pipeline complete")
    return final_df