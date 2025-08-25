# Auto-install plotly if not available
import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)

# Install required packages
install_and_import('plotly')

# Now import everything normally
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Multi-Model Evaluation Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(108, 99, 255, 0.1);
        border-radius: 10px;
    }
    
    .upload-section {
        background: rgba(108, 99, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(108, 99, 255, 0.2);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Model configurations
MODEL_COLORS = {
    'MODEL A': '#FF6B6B',
    'MODEL B': '#4ECDC4', 
    'MODEL C': '#45B7D1',
    'MODEL D': '#FECA57',
    'MODEL E': '#FF9FF3',
    'MODEL F': '#8B5CF6',
    'MODEL G': '#F59E0B'
}

MODEL_NAMES = {
    'MODEL A': 'LLAMA 3.1 8B INSTRUCT',
    'MODEL B': 'V1_INSTRUCT_SFT_CK34',
    'MODEL C': 'V2_BASE_CPT_SFT_CK21',
    'MODEL D': 'V2_BASE_CPT_SFT_DPO_RUN1',
    'MODEL E': 'V2_BASE_CPT_SFT_DPO_RUN2',
    'MODEL F': 'V2_BASE_CPT_RESIDUAL',
    'MODEL G': 'V2_BASE_CPT_RESIDUAL_CONCISE'
}

# Column mappings
COLUMN_MAPPINGS = {
    'qa': {
        'judge_columns': [
            'Judge_Model_A_Score',
            'Judge_Model_B_Score',
            'Judge_Model_C_Score',
            'Judge_Model_F_Score',
            'Judge_Model_G_Score',
            'Judge_Model_H_Score',
            'Judge_Model_I_Score'
        ],
        'bert_columns': [
            'f1_base',
            'f1_V34',
            'bertscore_f1_v21',
            'bertscore_f1_v2_dpo_run1',
            'bertscore_f1_v2_dpo_run2',
            'bertscore_f1_v2_cpt_residual',
            'bertscore_f1_V2_BASE_CPT_RESIDUAL_CONCISE_qa'
        ]
    },
    'summary': {
        'judge_columns': [
            'Judge_Model_A_Score',
            'Judge_Model_B_Score',
            'Judge_Model_C_Score',
            'Judge_Model_F_Score',
            'Judge_Model_G_Score',
            'Judge_Model_H_Score'
        ],
        'bert_columns': [
            'instruct_bertscore_f1',
            'finetune_bertscore_f1',
            'sft_v21_bertscore_f1',
            'bertscore_f1_v2_dpo_run1',
            'bertscore_f1_v2_dpo_run2',
            'bertscore_f1_v2_cpt_residual'
        ]
    },
    'classification': {
        'judge_columns': [
            'Judge_Model_A_Score',
            'Judge_Model_B_Score',
            'Judge_Model_C_Score',
            'Judge_Model_F_Score',
            'Judge_Model_G_Score',
            'Judge_Model_H_Score'
        ],
        'bert_columns': [
            'instruct_bertscore_f1',
            'finetune_bertscore_f1',
            'sft_v21_bertscore_f1',
            'bertscore_f1_v2_dpo_run1',
            'bertscore_f1_v2_dpo_run2',
            'bertscore_f1_v2_cpt_residual'
        ]
    }
}

def load_excel_data(uploaded_file):
    """Load and process Excel data"""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def calculate_averages(df, columns):
    """Calculate average scores for given columns"""
    averages = []
    for col in columns:
        if col in df.columns:
            valid_scores = pd.to_numeric(df[col], errors='coerce').dropna()
            if not valid_scores.empty:
                averages.append(valid_scores.mean())
            else:
                averages.append(0)
        else:
            averages.append(0)
    return averages

def create_judge_comparison_chart(datasets, task_filter=None):
    """Create judge scores comparison chart"""
    fig = go.Figure()
    
    tasks_to_show = [task_filter] if task_filter else list(datasets.keys())
    
    # Determine max models across tasks
    max_models = 0
    for task in tasks_to_show:
        if datasets[task] is not None:
            max_models = max(max_models, len(COLUMN_MAPPINGS[task]['judge_columns']))
    
    model_labels = [f"{model} ({MODEL_NAMES[model][:20]}...)" for model in list(MODEL_COLORS.keys())[:max_models]]
    
    for task in tasks_to_show:
        if datasets[task] is not None:
            df = datasets[task]
            judge_cols = COLUMN_MAPPINGS[task]['judge_columns']
            averages = calculate_averages(df, judge_cols)
            
            # Pad with zeros if needed
            while len(averages) < max_models:
                averages.append(0)
            
            fig.add_trace(go.Bar(
                name=task.title(),
                x=model_labels,
                y=averages,
                marker_color={'qa': '#FF6B6B', 'summary': '#4ECDC4', 'classification': '#45B7D1'}[task],
                opacity=0.8
            ))
    
    fig.update_layout(
        title="🏆 Judge Scores Comparison (1-5 Scale)",
        xaxis_title="Models",
        yaxis_title="Judge Score (1-5 Scale)",
        yaxis=dict(range=[1, 5]),
        showlegend=not task_filter,
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_bert_comparison_chart(datasets, task_filter=None):
    """Create BERT F1 scores comparison chart"""
    fig = go.Figure()
    
    tasks_to_show = [task_filter] if task_filter else list(datasets.keys())
    
    # Determine max models across tasks
    max_models = 0
    for task in tasks_to_show:
        if datasets[task] is not None:
            max_models = max(max_models, len(COLUMN_MAPPINGS[task]['bert_columns']))
    
    model_labels = [f"{model} ({MODEL_NAMES[model][:20]}...)" for model in list(MODEL_COLORS.keys())[:max_models]]
    
    for task in tasks_to_show:
        if datasets[task] is not None:
            df = datasets[task]
            bert_cols = COLUMN_MAPPINGS[task]['bert_columns']
            averages = calculate_averages(df, bert_cols)
            
            # Pad with zeros if needed
            while len(averages) < max_models:
                averages.append(0)
            
            fig.add_trace(go.Scatter(
                name=task.title(),
                x=model_labels,
                y=averages,
                mode='lines+markers',
                line=dict(
                    color={'qa': '#96CEB4', 'summary': '#FF9F43', 'classification': '#9966FF'}[task],
                    width=3
                ),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title="📊 BERT F1 Scores",
        xaxis_title="Models",
        yaxis_title="BERT F1 Score",
        yaxis=dict(range=[0, 1]),
        showlegend=not task_filter,
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_distribution_chart(datasets, task_filter=None):
    """Create judge score distribution chart"""
    all_scores = []
    
    tasks_to_show = [task_filter] if task_filter else list(datasets.keys())
    
    for task in tasks_to_show:
        if datasets[task] is not None:
            df = datasets[task]
            judge_cols = COLUMN_MAPPINGS[task]['judge_columns']
            
            for col in judge_cols:
                if col in df.columns:
                    scores = pd.to_numeric(df[col], errors='coerce').dropna()
                    scores = scores[(scores >= 1) & (scores <= 5)]
                    all_scores.extend(scores)
    
    if not all_scores:
        return go.Figure()
    
    # Calculate distribution
    distribution = [sum(1 for s in all_scores if round(s) == score) for score in range(1, 6)]
    
    fig = go.Figure(data=[go.Pie(
        labels=['Score 1 (Poor)', 'Score 2 (Below Avg)', 'Score 3 (Average)', 'Score 4 (Good)', 'Score 5 (Excellent)'],
        values=distribution,
        hole=.3,
        marker=dict(colors=['#FF6384', '#FF9F40', '#FFCD56', '#4BC0C0', '#36A2EB'])
    )])
    
    fig.update_layout(
        title="📈 Judge Score Distribution",
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_correlation_chart(datasets, task_filter=None):
    """Create Judge vs BERT correlation scatter plot"""
    fig = go.Figure()
    
    tasks_to_show = [task_filter] if task_filter else list(datasets.keys())
    
    for task in tasks_to_show:
        if datasets[task] is not None:
            df = datasets[task]
            mapping = COLUMN_MAPPINGS[task]
            
            for i, (judge_col, bert_col) in enumerate(zip(mapping['judge_columns'], mapping['bert_columns'])):
                if judge_col in df.columns and bert_col in df.columns:
                    judge_scores = pd.to_numeric(df[judge_col], errors='coerce')
                    bert_scores = pd.to_numeric(df[bert_col], errors='coerce')
                    
                    valid_mask = (~judge_scores.isna()) & (~bert_scores.isna()) & (judge_scores >= 1) & (judge_scores <= 5) & (bert_scores >= 0)
                    
                    if valid_mask.sum() > 0:
                        model_key = list(MODEL_COLORS.keys())[i]
                        fig.add_trace(go.Scatter(
                            x=bert_scores[valid_mask],
                            y=judge_scores[valid_mask],
                            mode='markers',
                            name=f"{task} - {model_key}",
                            marker=dict(
                                color=MODEL_COLORS[model_key],
                                size=8,
                                opacity=0.7
                            )
                        ))
    
    fig.update_layout(
        title="⚖️ Judge vs BERT Correlation",
        xaxis_title="BERT F1 Score",
        yaxis_title="Judge Score (1-5)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[1, 5]),
        height=500,
        template="plotly_white",
        showlegend=not task_filter
    )
    
    return fig

def create_task_comparison_chart(datasets):
    """Create task performance comparison chart"""
    fig = go.Figure()
    
    valid_tasks = [task for task, data in datasets.items() if data is not None]
    if not valid_tasks:
        return go.Figure()
    
    # Determine max models
    max_models = max(len(COLUMN_MAPPINGS[task]['judge_columns']) for task in valid_tasks)
    
    for i in range(max_models):
        model_key = list(MODEL_COLORS.keys())[i]
        task_scores = []
        
        for task in valid_tasks:
            df = datasets[task]
            judge_cols = COLUMN_MAPPINGS[task]['judge_columns']
            
            if i < len(judge_cols) and judge_cols[i] in df.columns:
                scores = pd.to_numeric(df[judge_cols[i]], errors='coerce').dropna()
                scores = scores[(scores >= 1) & (scores <= 5)]
                avg_score = scores.mean() if not scores.empty else 0
            else:
                avg_score = 0
                
            task_scores.append(avg_score)
        
        fig.add_trace(go.Bar(
            name=model_key,
            x=[task.title() for task in valid_tasks],
            y=task_scores,
            marker_color=MODEL_COLORS[model_key],
            opacity=0.8
        ))
    
    fig.update_layout(
        title="🎯 Task Performance Comparison",
        xaxis_title="Tasks",
        yaxis_title="Average Judge Score (1-5 Scale)",
        yaxis=dict(range=[1, 5]),
        height=500,
        template="plotly_white",
        barmode='group'
    )
    
    return fig

def calculate_best_model(datasets):
    """Calculate the best overall model"""
    model_scores = {model: [] for model in MODEL_COLORS.keys()}
    
    for task, df in datasets.items():
        if df is not None:
            judge_cols = COLUMN_MAPPINGS[task]['judge_columns']
            
            for i, col in enumerate(judge_cols):
                if col in df.columns and i < len(model_scores):
                    model_key = list(MODEL_COLORS.keys())[i]
                    scores = pd.to_numeric(df[col], errors='coerce').dropna()
                    scores = scores[(scores >= 1) & (scores <= 5)]
                    if not scores.empty:
                        model_scores[model_key].extend(scores.tolist())
    
    best_model = 'MODEL A'
    best_score = 0
    
    for model, scores in model_scores.items():
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
    
    return best_model, best_score

# Main Streamlit App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Multi-Model Evaluation Dashboard</h1>
        <p>Comprehensive Analysis with Judge Scores (1-5 Scale) & BERT F1 Scores</p>
        <p><em>QnA: 7 Models | Summary & Classification: 6 Models</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Legend
    st.subheader("🎯 Model Legend")
    cols = st.columns(2)
    
    for i, (model, name) in enumerate(MODEL_NAMES.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 20px; height: 20px; background-color: {MODEL_COLORS[model]}; 
                     border-radius: 50%; margin-right: 10px;"></div>
                <strong>{model}:</strong> {name}
            </div>
            """, unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📂 Upload Dataset Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qa_file = st.file_uploader("📊 QA Dataset (Excel)", type=['xlsx', 'xls'], key="qa")
    
    with col2:
        summary_file = st.file_uploader("📝 Summary Dataset (Excel)", type=['xlsx', 'xls'], key="summary")
    
    with col3:
        classification_file = st.file_uploader("🏷️ Classification Dataset (Excel)", type=['xlsx', 'xls'], key="classification")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load datasets
    datasets = {
        'qa': load_excel_data(qa_file) if qa_file else None,
        'summary': load_excel_data(summary_file) if summary_file else None,
        'classification': load_excel_data(classification_file) if classification_file else None
    }
    
    # Summary Statistics
    if any(df is not None for df in datasets.values()):
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        tasks_loaded = sum(1 for df in datasets.values() if df is not None)
        total_samples = sum(len(df) for df in datasets.values() if df is not None)
        best_model, best_score = calculate_best_model(datasets)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Samples</h4>
                <div style="font-size: 2rem; font-weight: bold;">{total_samples:,}</div>
                <div>Across All Tasks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Tasks Loaded</h4>
                <div style="font-size: 2rem; font-weight: bold;">{tasks_loaded}</div>
                <div>Out of 3</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Best Overall Model</h4>
                <div style="font-size: 2rem; font-weight: bold;">{best_model}</div>
                <div>Score: {best_score:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Task Filter
    if any(df is not None for df in datasets.values()):
        st.subheader("🎛️ Analysis Controls")
        
        available_tasks = ['overview'] + [task for task, df in datasets.items() if df is not None]
        selected_task = st.selectbox(
            "Select Task for Analysis",
            options=available_tasks,
            format_func=lambda x: x.title() if x != 'overview' else '📈 Overview'
        )
        
        task_filter = None if selected_task == 'overview' else selected_task
        
        # Charts
        st.subheader("📈 Visualization Dashboard")
        
        if task_filter:
            st.info(f"Showing analysis for: **{task_filter.title()}** task")
        else:
            st.info("Showing overview across all loaded tasks")
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_judge_comparison_chart(datasets, task_filter)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig3 = create_distribution_chart(datasets, task_filter)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig2 = create_bert_comparison_chart(datasets, task_filter)
            st.plotly_chart(fig2, use_container_width=True)
            
            fig4 = create_correlation_chart(datasets, task_filter)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Full-width task comparison (only for overview)
        if not task_filter:
            fig5 = create_task_comparison_chart(datasets)
            st.plotly_chart(fig5, use_container_width=True)
        
        # Data Preview Section
        if task_filter and datasets[task_filter] is not None:
            st.subheader(f"📋 {task_filter.title()} Data Preview")
            df = datasets[task_filter]
            
            # Show basic info
            st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
            
            # Show relevant columns
            judge_cols = COLUMN_MAPPINGS[task_filter]['judge_columns']
            bert_cols = COLUMN_MAPPINGS[task_filter]['bert_columns']
            
            available_judge_cols = [col for col in judge_cols if col in df.columns]
            available_bert_cols = [col for col in bert_cols if col in df.columns]
            
            if available_judge_cols or available_bert_cols:
                preview_cols = available_judge_cols[:5] + available_bert_cols[:5]  # Show first 5 of each
                st.dataframe(df[preview_cols].head(10))
            else:
                st.warning("No relevant columns found for this task.")
    
    else:
        st.info("👆 Please upload at least one Excel file to start the analysis.")
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **For each task, your Excel file should contain:**
            
            **QA Task:**
            - Judge score columns: `Judge_Model_A_Score`, `Judge_Model_B_Score`, etc.
            - BERT F1 columns: `f1_base`, `f1_V34`, `bertscore_f1_v21`, etc.
            
            **Summary Task:**
            - Judge score columns: `Judge_Model_A_Score`, `Judge_Model_B_Score`, etc.
            - BERT F1 columns: `instruct_bertscore_f1`, `finetune_bertscore_f1`, etc.
            
            **Classification Task:**
            - Same structure as Summary task
            
            All judge scores should be on a 1-5 scale, and BERT F1 scores should be between 0-1.
            """)

if __name__ == "__main__":
    main()