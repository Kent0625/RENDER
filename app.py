import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.datasets import make_classification, make_regression

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Evaluation Metrics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def plot_confusion_matrix_heatmap(cm, labels, title="Confusion Matrix"):
    """
    Draws a heatmap using Seaborn/Matplotlib for the given confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(title)
    return fig

def generate_binary_cm_inputs(key_prefix="bin"):
    """
    Creates sliders for TP, TN, FP, FN for Binary Classification.
    Returns the constructed Confusion Matrix (2x2).
    """
    st.markdown("### 🎚️ Adjust Matrix Values")
    col1, col2 = st.columns(2)
    with col1:
        tp = st.slider("True Positives (TP)", 0, 100, 50, key=f"{key_prefix}_tp")
        fn = st.slider("False Negatives (FN)", 0, 100, 10, key=f"{key_prefix}_fn")
    with col2:
        fp = st.slider("False Positives (FP)", 0, 100, 10, key=f"{key_prefix}_fp")
        tn = st.slider("True Negatives (TN)", 0, 100, 50, key=f"{key_prefix}_tn")
    
    # Structure: [[TN, FP], [FN, TP]] is standard for sklearn if labels=[0, 1]
    # But usually visualizers expect:
    #      Pred 0  Pred 1
    # Act 0   TN      FP
    # Act 1   FN      TP
    cm = np.array([[tn, fp], [fn, tp]])
    return cm, ["Negative", "Positive"]

def generate_multiclass_cm_inputs(key_prefix="multi"):
    """
    Simulates a Multi-class confusion matrix based on sample size and noise.
    Returns the CM and labels.
    """
    st.markdown("### 🎚️ Simulation Settings")
    n_samples = st.slider("Number of Samples", 50, 1000, 300, step=50, key=f"{key_prefix}_n")
    noise = st.slider("Noise Level (Confusion Probability)", 0.0, 1.0, 0.2, key=f"{key_prefix}_noise")
    
    # Generate synthetic data
    X, y_true = make_classification(
        n_samples=n_samples, 
        n_features=5, 
        n_informative=3, 
        n_classes=3, 
        flip_y=noise, 
        random_state=42
    )
    # Simulate predictions (add some random noise to truth to create errors)
    # Simple logic: with probability (1-noise), y_pred = y_true, else random
    y_pred = y_true.copy()
    mask = np.random.rand(n_samples) < noise
    y_pred[mask] = np.random.randint(0, 3, size=mask.sum())
    
    labels = ["Cat", "Dog", "Bird"]
    cm = confusion_matrix(y_true, y_pred)
    return cm, labels, y_true, y_pred

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("ML Evaluation Dashboard")
st.sidebar.info("Select a module to explore metrics.")

module = st.sidebar.radio(
    "Go to Module:",
    [
        "1. Accuracy",
        "2. Precision & Recall",
        "3. F1-Score & Specificity",
        "4. ROC/AUC & KS Chart",
        "5. Regression Metrics",
        "6. Correlation"
    ]
)

# -----------------------------------------------------------------------------
# MODULE 1: ACCURACY
# -----------------------------------------------------------------------------
if module == "1. Accuracy":
    st.title("🎯 Module 1: Accuracy")
    st.markdown("Accuracy measures how often the classifier is correct overall.")
    
    mode = st.radio("Classification Type:", ["Binary (2 Classes)", "Multi-class (3 Classes)"], horizontal=True)
    
    if mode == "Binary (2 Classes)":
        cm, labels = generate_binary_cm_inputs("acc")
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        accuracy = (tp + tn) / total if total > 0 else 0
        
        col_res, col_vis = st.columns([1, 2])
        with col_res:
            st.metric("Accuracy", f"{accuracy:.2%}")
            st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
            st.write(f"Calculation: ({tp} + {tn}) / {total}")
            
        with col_vis:
            st.pyplot(plot_confusion_matrix_heatmap(cm, labels))
            
    else:
        cm, labels, y_true, y_pred = generate_multiclass_cm_inputs("acc_multi")
        acc = accuracy_score(y_true, y_pred)
        
        col_res, col_vis = st.columns([1, 2])
        with col_res:
            st.metric("Global Accuracy", f"{acc:.2%}")
            st.markdown("**Note:** In multi-class, accuracy is the fraction of correct predictions (diagonal sum) over total samples.")
        
        with col_vis:
            st.pyplot(plot_confusion_matrix_heatmap(cm, labels))

# -----------------------------------------------------------------------------
# MODULE 2: PRECISION & RECALL
# -----------------------------------------------------------------------------
elif module == "2. Precision & Recall":
    st.title("🔍 Module 2: Precision & Recall")
    st.markdown("""
    - **Precision**: Of all positive predictions, how many were actually positive? (Quality)
    - **Recall (Sensitivity)**: Of all actual positives, how many did we find? (Quantity)
    """ )
    
    mode = st.radio("Classification Type:", ["Binary (2 Classes)", "Multi-class (3 Classes)"], horizontal=True, key="pr_mode")
    
    if mode == "Binary (2 Classes)":
        cm, labels = generate_binary_cm_inputs("pr")
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        col_res, col_vis = st.columns([1, 2])
        with col_res:
            st.metric("Precision", f"{precision:.2f}")
            st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
            
            st.divider()
            
            st.metric("Recall", f"{recall:.2f}")
            st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
            
        with col_vis:
            st.pyplot(plot_confusion_matrix_heatmap(cm, labels))
            
    else:
        cm, labels, y_true, y_pred = generate_multiclass_cm_inputs("pr_multi")
        
        # Calculate per-class metrics
        prec_per_class = precision_score(y_true, y_pred, average=None)
        rec_per_class = recall_score(y_true, y_pred, average=None)
        
        col_res, col_vis = st.columns([1, 2])
        with col_res:
            st.markdown("### Per-Class Metrics")
            res_df = pd.DataFrame({
                "Class": labels,
                "Precision": prec_per_class,
                "Recall": rec_per_class
            })
            st.dataframe(res_df.style.format("{:.2f}"))
            
            st.markdown("**Macro Average:** Unweighted mean of metrics.")
            st.write(f"Macro Precision: {precision_score(y_true, y_pred, average='macro'):.2f}")
            st.write(f"Macro Recall: {recall_score(y_true, y_pred, average='macro'):.2f}")
            
        with col_vis:
            st.pyplot(plot_confusion_matrix_heatmap(cm, labels))

# -----------------------------------------------------------------------------
# MODULE 3: F1-SCORE & SPECIFICITY
# -----------------------------------------------------------------------------
elif module == "3. F1-Score & Specificity":
    st.title("⚖️ Module 3: F1-Score & Specificity")
    st.markdown("""
    - **F1-Score**: Harmonic mean of Precision and Recall. Good for imbalanced datasets.
    - **Specificity**: True Negative Rate. How well do we avoid False Positives?
    """ )
    
    mode = st.radio("Classification Type:", ["Binary (2 Classes)", "Multi-class (3 Classes)"], horizontal=True, key="f1_mode")
    
    if mode == "Binary (2 Classes)":
        cm, labels = generate_binary_cm_inputs("f1")
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        col_res, col_vis = st.columns([1, 2])
        with col_res:
            st.metric("F1-Score", f"{f1:.2f}")
            st.latex(r"F1 = 2 \times \frac{P \times R}{P + R}")
            
            st.divider()
            
            st.metric("Specificity", f"{specificity:.2f}")
            st.latex(r"\text{Specificity} = \frac{TN}{TN + FP}")
            
        with col_vis:
            st.pyplot(plot_confusion_matrix_heatmap(cm, labels))
            
    else:
        cm, labels, y_true, y_pred = generate_multiclass_cm_inputs("f1_multi")
        
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Specificity for multiclass is usually One-vs-Rest calculation
        # We calculate it manually for each class
        specificities = []
        for i in range(len(labels)):
            # Treat class i as Positive, others as Negative
            # cm[i, i] is TP for class i
            # sum(cm[:, i]) is TP + FP (Predicted Positive)
            # sum(cm[i, :]) is TP + FN (Actual Positive)
            # Total sum is Total
            
            # Simple One-vs-Rest Logic from Confusion Matrix:
            # TN_i = Total - (Row_i + Col_i - Cell_ii)
            # FP_i = Col_i - Cell_ii
            
            total = cm.sum()
            tp_i = cm[i, i]
            fp_i = cm[:, i].sum() - tp_i
            fn_i = cm[i, :].sum() - tp_i
            tn_i = total - (tp_i + fp_i + fn_i)
            
            spec_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
            specificities.append(spec_i)
            
        col_res, col_vis = st.columns([1, 2])
        with col_res:
            st.metric("Macro F1-Score", f"{f1_macro:.2f}")
            
            st.markdown("### Per-Class Specificity")
            spec_df = pd.DataFrame({
                "Class": labels,
                "Specificity": specificities
            })
            st.dataframe(spec_df.style.format("{:.2f}"))
            
        with col_vis:
            st.pyplot(plot_confusion_matrix_heatmap(cm, labels))

# -----------------------------------------------------------------------------
# MODULE 4: ROC/AUC & KS CHART
# -----------------------------------------------------------------------------
elif module == "4. ROC/AUC & KS Chart":
    st.title("📈 Module 4: ROC/AUC & KS Chart")
    st.markdown("Explore how the threshold affects True Positive and False Positive Rates.")
    
    col_input, col_viz = st.columns([1, 3])
    
    with col_input:
        st.subheader("Settings")
        separation = st.slider("Class Separation (Difficulty)", 0.1, 5.0, 2.0, help="Higher means easier to separate.")
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5)
        
    # Generate Synthetic Data for Probabilities
    # Class 0: Gaussian centered at 0
    # Class 1: Gaussian centered at 'separation'
    np.random.seed(42)
    n_per_class = 1000
    noise_scale = 1.5
    
    # Scores (logits-like)
    neg_scores = np.random.normal(0, noise_scale, n_per_class)
    pos_scores = np.random.normal(separation, noise_scale, n_per_class)
    
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    scores = np.concatenate([neg_scores, pos_scores])
    
    # Min-max scale scores to 0-1 probability range
    scores_prob = (scores - scores.min()) / (scores.max() - scores.min())
    
    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(y, scores_prob)
    roc_auc = auc(fpr, tpr)
    
    # Current point on ROC based on slider threshold
    # Find index of threshold closest to slider value
    closest_idx = (np.abs(thresholds - threshold)).argmin()
    current_fpr = fpr[closest_idx]
    current_tpr = tpr[closest_idx]
    
    # --- VISUALIZATION ---
    with col_viz:
        tab1, tab2 = st.tabs(["ROC Curve", "KS Chart & Distributions"])
        
        with tab1:
            fig_roc = px.area(
                x=fpr, y=tpr, 
                title=f'ROC Curve (AUC = {roc_auc:.3f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
            )
            fig_roc.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            # Add marker for current threshold
            fig_roc.add_trace(
                go.Scatter(
                    x=[current_fpr], y=[current_tpr],
                    mode='markers+text',
                    marker=dict(color='red', size=12),
                    text=[f'Thresh={threshold:.2f}'],
                    textposition="bottom right",
                    name='Current Threshold'
                )
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with tab2:
            # KS Statistic is max separation between CDFs
            # But visually usually shown as separation between PDFs or distance in CDFs
            
            # Create DataFrame for histogram
            hist_df = pd.DataFrame({
                "Probability Score": scores_prob,
                "True Class": ["Negative" if x==0 else "Positive" for x in y]
            })
            
            fig_dist = px.histogram(
                hist_df, x="Probability Score", color="True Class",
                nbins=50, opacity=0.6, marginal="box",
                title="Class Probability Distributions"
            )
            fig_dist.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # KS Stat Calculation
            # Sort scores
            desc_score_indices = np.argsort(scores_prob)[::-1]
            y_sorted = y[desc_score_indices]
            prob_sorted = scores_prob[desc_score_indices]
            
            # Calculate CDFs
            n_pos = sum(y)
            n_neg = len(y) - n_pos
            tp_cumsum = np.cumsum(y_sorted) / n_pos
            fp_cumsum = np.cumsum(1 - y_sorted) / n_neg
            
            ks_stat = np.max(np.abs(tp_cumsum - fp_cumsum))
            st.metric("KS Statistic", f"{ks_stat:.3f}", help="Max distance between TPR and FPR cumulative distributions.")

# -----------------------------------------------------------------------------
# MODULE 5: REGRESSION METRICS
# -----------------------------------------------------------------------------
elif module == "5. Regression Metrics":
    st.title("📉 Module 5: Regression Metrics")
    st.markdown("Visualize residuals and how errors impact RMSE, MSE, and MAE.")
    
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        noise_level = st.slider("Add Noise (Error)", 0.0, 50.0, 10.0)
        n_points = 100
        
    # Generate Linear Data
    np.random.seed(42)
    X = np.linspace(0, 100, n_points)
    true_slope = 2.5
    true_intercept = 10
    
    y_perfect = true_slope * X + true_intercept
    y_actual = y_perfect + np.random.normal(0, noise_level, n_points)
    
    # Simple calculation (Assuming we predicted the perfect line, or we fit a line?)
    # Usually in teaching, we show how the 'fit' degrades or just calculate metrics against the 'perfect' line as 'prediction'
    # Let's say our "Model" predicts the perfect line (underlying trend) and we measure the error of the noisy data against it.
    # OR we fit a line to the noisy data. Let's fit a line to show realistic residuals.
    
    # Fit line
    m, c = np.polyfit(X, y_actual, 1)
    y_pred = m * X + c
    
    # Metrics
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    
    with col_input:
        st.divider()
        st.metric("RMSE (Root Mean Sq Error)", f"{rmse:.2f}")
        st.metric("MSE (Mean Squared Error)", f"{mse:.2f}")
        st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        
    with col_viz:
        fig_reg = go.Figure()
        
        # Scatter actual
        fig_reg.add_trace(go.Scatter(
            x=X, y=y_actual, mode='markers', name='Actual Data',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Regression Line
        fig_reg.add_trace(go.Scatter(
            x=X, y=y_pred, mode='lines', name='Predicted (Regression Line)',
            line=dict(color='red', width=3)
        ))
        
        # Draw Residual lines (errors)
        # We draw lines from y_actual to y_pred
        # For performance, maybe just draw a few or standard error bars
        # Let's draw lines for all points using shapes or a loop
        # A clearer way in plotly is to add line segments.
        
        # Adding a visual cue for residuals (showing only first 20 for clarity if needed, but let's try all faint)
        x_lines = []
        y_lines = []
        for xi, yi_act, yi_pred in zip(X, y_actual, y_pred):
            x_lines.extend([xi, xi, None])
            y_lines.extend([yi_act, yi_pred, None])
            
        fig_reg.add_trace(go.Scatter(
            x=x_lines, y=y_lines,
            mode='lines',
            name='Residuals (Errors)',
            line=dict(color='gray', width=1, dash='dot'),
            hoverinfo='skip'
        ))
        
        fig_reg.update_layout(title="Regression Fit & Residuals", xaxis_title="X", yaxis_title="y")
        st.plotly_chart(fig_reg, use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE 6: CORRELATION
# -----------------------------------------------------------------------------
elif module == "6. Correlation":
    st.title("🔗 Module 6: Correlation")
    st.markdown("Pearson's R and R-Squared: Measuring strength of linear relationships.")
    
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        r_target = st.slider("Target Correlation (Pearson's R)", -1.0, 1.0, 0.85, step=0.05)
        n_samples = 200
        
    # Generate Correlated Data using Covariance Matrix
    # Covariance = Correlation * std_x * std_y
    # We'll assume std_x = 1, std_y = 1
    
    mean = [0, 0]
    cov = [[1, r_target], [r_target, 1]]
    
    try:
        data = np.random.multivariate_normal(mean, cov, n_samples)
        x_corr = data[:, 0]
        y_corr = data[:, 1]
    except:
        # Fallback if matrix is not positive semi-definite (rare with this simple setup unless r=1/-1 exactly)
        x_corr = np.random.normal(0, 1, n_samples)
        y_corr = x_corr * r_target + np.random.normal(0, 0.5, n_samples)
    
    # Calculate actual metrics from sample
    df_corr = pd.DataFrame({'X': x_corr, 'Y': y_corr})
    actual_r = df_corr.corr().iloc[0, 1]
    r_squared = actual_r ** 2
    
    with col_input:
        st.divider()
        st.metric("Pearson's R", f"{actual_r:.3f}")
        st.metric("R-Squared (R²)", f"{r_squared:.3f}")
        st.info("Note: R² represents the proportion of variance for a dependent variable that's explained by an independent variable.")
        
    with col_viz:
        fig_corr = px.scatter(
            df_corr, x='X', y='Y', 
            trendline="ols", 
            title=f"Correlation Scatter Plot (R = {actual_r:.2f})",
            trendline_color_override="red"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# DEPLOYMENT INFO
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Deployment")
st.sidebar.caption("1. Push to GitHub.")
st.sidebar.caption("2. New Web Service on Render.")
st.sidebar.caption("3. Build: `pip install -r requirements.txt`")
st.sidebar.caption("4. Start: `streamlit run app.py --server.port $PORT`")
