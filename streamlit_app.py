import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Humanitarian Impact Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and preprocessing objects
@st.cache_resource
@st.cache_resource
def load_model():
    """Load all model components from pickle files"""
    try:
        model = joblib.load('models/humanitarian_impact_model.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
        target_encoder = joblib.load('models/target_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        metadata = joblib.load('models/model_metadata.pkl')
        mappings = joblib.load('models/label_mappings.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, encoders, target_encoder, feature_names, metadata, mappings, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found! Please run the training script first.\n\nMissing file: {e}")
        st.stop()

# Load everything
model, encoders, target_encoder, feature_names, metadata, mappings, scaler = load_model()

# ============================================================================
# SIDEBAR - Information
# ============================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f30d.png", width=80)
    st.title("About This Model")
    
    st.markdown("""
    ### 🎯 What It Predicts
    This model predicts **Humanitarian Impact Level** based on multiple factors:
    
    **🟢 Limited Impact**
    - Minimal humanitarian needs
    
    **🟡 Moderate Impact**
    - Standard humanitarian response
    
    **🟠 Severe Impact**
    - Significant humanitarian crisis
    
    **🔴 Critical Impact**
    - Major humanitarian emergency
    
    ---
    
    ### 📊 Model Performance
    """)
    
    st.metric("Model Type", metadata['model_name'])
    st.metric("Accuracy", f"{metadata['accuracy']:.1%}")
    st.metric("F1-Score", f"{metadata['f1_score']:.1%}")
    st.metric("CV Score", f"{metadata['cv_mean_score']:.1%}")
    
    st.markdown(f"""
    ---
    
    ### 📅 Model Info
    - **Trained on**: {metadata['train_date']}
    - **Data source**: UCDP Dataset
    - **Target**: Multi-factor impact assessment
    
    ---
    
    ### 💡 Key Features
    This model uses **multiple factors** beyond just death counts:
    - Conflict type & region
    - Death estimates & uncertainty
    - Temporal patterns
    - Regional instability factors
    
    ---
    
    ### ⚠️ Important Note
    This is an educational tool. Predictions should support, not replace, expert analysis.
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.title("🌍 Humanitarian Impact Prediction System")
st.markdown("""
This machine learning system predicts the **humanitarian impact level** of armed conflicts 
based on multiple factors including conflict characteristics, casualty estimates, regional context, and uncertainty.

**Note**: This model does NOT simply classify by death count thresholds. It uses machine learning to identify 
patterns across multiple features to assess overall humanitarian impact.
""")

# Add tabs for different sections
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📚 How It Works", "📈 Model Insights"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.header("Enter Conflict Information")

    # UCDP conflict type labels (hardcoded - always reliable)
    conflict_type_map = {
        1: "Extrasystemic conflict",
        2: "Interstate conflict",
        3: "Intrastate conflict",
        4: "Internationalized intrastate conflict"
    }

    # UCDP region labels (hardcoded - always reliable)
    region_map = {
        1: "Europe",
        2: "Middle East",
        3: "Asia",
        4: "Africa",
        5: "Americas"
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location & Type")
        
        # UCDP conflict type labels (hardcoded - always reliable)
        conflict_type_map = {
            1: "Extrasystemic conflict",
            2: "Interstate conflict",
            3: "Intrastate conflict",
            4: "Internationalized intrastate conflict"
        }
        
        # UCDP region labels (hardcoded - always reliable)
        region_map = {
            1: "Europe",
            2: "Middle East",
            3: "Asia",
            4: "Africa",
            5: "America"
        }
        
        # Conflict Type
        conflict_type = st.selectbox(
            "Conflict Type",
            options=list(conflict_type_map.keys()),
            format_func=lambda x: conflict_type_map[x]
        )
        
        # Region
        region = st.selectbox(
            "Region",
            options=list(region_map.keys()),
            format_func=lambda x: region_map[x]
        )
        
        # Year
        year = st.number_input(
            "Year",
            min_value=1989,
            max_value=2030,
            value=2024,
            help="Year of the conflict"
        )
    
    with col2:
        st.subheader("💀 Casualty Estimates")
        
        bd_low = st.number_input(
            "Battle Deaths (Low Estimate)",
            min_value=0,
            max_value=100000,
            value=100,
            step=50,
            help="Lower bound estimate of battle-related deaths"
        )
        
        bd_high = st.number_input(
            "Battle Deaths (High Estimate)",
            min_value=0,
            max_value=100000,
            value=500,
            step=50,
            help="Upper bound estimate of battle-related deaths"
        )
        
        if bd_high < bd_low:
            st.warning("⚠️ High estimate should be greater than low estimate")
        
        bd_estimate = (bd_low + bd_high) / 2
        st.info(f"📊 Average Estimate: **{bd_estimate:.0f}** deaths")
    
    with col3:
        st.subheader("📊 Impact Factors")
        
        st.markdown("""
        **Model considers:**
        
        ✓ Conflict type & region
        
        ✓ Death estimates (low/high)
        
        ✓ Uncertainty in estimates
        
        ✓ Temporal patterns
        
        ✓ Regional context
        
        ---
        
        **Not just death counts!**
        
        The model learns complex patterns across all factors.
        """)
    
    # Calculate derived features (same as training)
    death_range = bd_high - bd_low
    uncertainty_ratio = death_range / (bd_estimate + 1)
    log_death_estimate = np.log1p(bd_estimate)
    decade = (year // 10) * 10
    years_since_1989 = year - 1989
    
    # Show calculated features
    with st.expander("🔍 View Calculated Features"):
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        with feat_col1:
            st.metric("Death Range", f"{death_range:.0f}")
            st.metric("Uncertainty Ratio", f"{uncertainty_ratio:.3f}")
        with feat_col2:
            st.metric("Log(Deaths)", f"{log_death_estimate:.3f}")
            st.metric("Decade", decade)
        with feat_col3:
            st.metric("Years Since 1989", years_since_1989)
    
    # Create input dataframe (matching training features exactly)
    input_data = pd.DataFrame({
        'type_of_conflict': [conflict_type],
        'region': [region],
        'year': [year],
        'bd_low': [bd_low],
        'bd_high': [bd_high],
        'death_range': [death_range],
        'uncertainty_ratio': [uncertainty_ratio],
        'decade': [decade],
        'years_since_1989': [years_since_1989],
        'log_death_estimate': [log_death_estimate]
    })
    
    # Ensure correct column order
    input_data = input_data[feature_names]
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 PREDICT HUMANITARIAN IMPACT", type="primary", use_container_width=True):
        with st.spinner("Analyzing conflict data using machine learning..."):
            
            # Check if model uses scaling
            uses_scaling = metadata.get('uses_scaling', False)
            
            # Make prediction
            if uses_scaling:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled) if hasattr(model, 'predict_proba') else None
            else:
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
            
            predicted_class = target_encoder.inverse_transform(prediction)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Show input summary
            # Show input summary
            st.info(f"""
            **Input Summary:**
            - **Conflict Type**: {conflict_type_map.get(conflict_type, 'Unknown')}
            - **Region**: {region_map.get(region, 'Unknown')}
            - **Year**: {year}
            - **Estimated Deaths**: {bd_estimate:.0f} (Range: {bd_low}-{bd_high})
            - **Uncertainty**: {uncertainty_ratio:.2f}
            """)
            
            st.markdown("---")
            
            # Main prediction with color coding
            if predicted_class == "Critical Impact":
                st.error(f"### 🔴 PREDICTED IMPACT: **{predicted_class.upper()}**")
                st.markdown("""
                **Interpretation**: This conflict is predicted to have **critical humanitarian impact** 
                requiring immediate large-scale international response.
                """)
            elif predicted_class == "Severe Impact":
                st.warning(f"### 🟠 PREDICTED IMPACT: **{predicted_class.upper()}**")
                st.markdown("""
                **Interpretation**: This conflict is predicted to have **severe humanitarian impact** 
                requiring significant humanitarian assistance.
                """)
            elif predicted_class == "Moderate Impact":
                st.info(f"### 🟡 PREDICTED IMPACT: **{predicted_class.upper()}**")
                st.markdown("""
                **Interpretation**: This conflict is predicted to have **moderate humanitarian impact** 
                requiring standard humanitarian response.
                """)
            else:  # Limited Impact
                st.success(f"### 🟢 PREDICTED IMPACT: **{predicted_class.upper()}**")
                st.markdown("""
                **Interpretation**: This conflict is predicted to have **limited humanitarian impact** 
                with minimal external assistance needed.
                """)
            
            st.markdown("---")
            
            # Probability breakdown
            if prediction_proba is not None:
                st.subheader("📈 Confidence Levels")
                
                prob_cols = st.columns(len(target_encoder.classes_))
                
                for i, class_name in enumerate(target_encoder.classes_):
                    prob = prediction_proba[0][i]
                    
                    with prob_cols[i]:
                        # Color code based on impact level
                        if "Critical" in class_name:
                            color = "🔴"
                        elif "Severe" in class_name:
                            color = "🟠"
                        elif "Moderate" in class_name:
                            color = "🟡"
                        else:
                            color = "🟢"
                        
                        st.metric(
                            label=f"{color} {class_name}",
                            value=f"{prob:.1%}"
                        )
                        st.progress(float(prob))
                
                # Confidence interpretation
                max_prob = max(prediction_proba[0])
                st.markdown("---")
                
                if max_prob > 0.8:
                    st.success("✅ **High Confidence**: The model is very confident in this prediction.")
                elif max_prob > 0.6:
                    st.info("ℹ️ **Moderate Confidence**: The model has reasonable confidence in this prediction.")
                else:
                    st.warning("⚠️ **Low Confidence**: The prediction is uncertain. Multiple impact levels are possible.")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommended Actions")
            
            if predicted_class == "Critical Impact":
                st.markdown("""
                **Immediate Actions Required:**
                - 🚨 Activate emergency response protocols
                - 🏥 Deploy large-scale medical teams and supplies
                - 🏕️ Prepare for mass displacement and refugee crisis
                - 🍽️ Establish emergency food distribution networks
                - 🤝 Coordinate international humanitarian response
                - 💰 Mobilize significant funding and resources
                - 📞 Establish 24/7 coordination centers
                """)
            elif predicted_class == "Severe Impact":
                st.markdown("""
                **Significant Response Needed:**
                - 🏥 Deploy medical teams and humanitarian aid
                - 🏕️ Prepare shelters and displacement support
                - 🍽️ Organize food and water distribution
                - 🤝 Coordinate with international organizations
                - 📊 Conduct rapid needs assessment
                - 👀 Monitor for potential escalation
                """)
            elif predicted_class == "Moderate Impact":
                st.markdown("""
                **Standard Humanitarian Response:**
                - 🏥 Provide medical assistance and supplies
                - 🏕️ Support local displacement efforts
                - 🍽️ Ensure food security measures
                - 📋 Maintain diplomatic and humanitarian channels
                - 📊 Regular monitoring and assessment
                - ⚠️ Watch for escalation indicators
                """)
            else:  # Limited Impact
                st.markdown("""
                **Monitoring and Prevention:**
                - 👁️ Observe and document the situation
                - 🕊️ Support conflict prevention initiatives
                - 🤝 Strengthen local mediation efforts
                - 📊 Monitor for signs of escalation
                - 🛡️ Early intervention if situation changes
                - 💬 Maintain communication channels
                """)
            
            # Feature influence explanation
            st.markdown("---")
            st.subheader("🔍 Understanding the Prediction")
            
            st.markdown("""
            **This prediction is based on:**
            
            1. **Multiple Factors** - Not just death estimates, but conflict type, region, uncertainty, and temporal patterns
            2. **Machine Learning Patterns** - The model learned from historical conflicts to identify risk patterns
            3. **Composite Assessment** - Combines quantitative and qualitative factors for holistic evaluation
            
            **Key Considerations:**
            - High uncertainty in estimates may increase predicted impact
            - Certain regions and conflict types historically have higher humanitarian needs
            - Recent conflicts may have different patterns than historical ones
            """)

# ============================================================================
# TAB 2: HOW IT WORKS
# ============================================================================
with tab2:
    st.header("📚 How This Model Works")
    
    st.markdown("""
    ### 🎯 Purpose
    
    This system predicts **humanitarian impact levels** using a multi-factor machine learning approach 
    that goes beyond simple death count thresholds.
    
    ---
    
    ### 📊 Impact Categories
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🟢 Limited Impact
        
        **Characteristics**:
        - Minimal humanitarian needs
        - Local response sufficient
        - Low regional instability
        
        #### 🟡 Moderate Impact
        
        **Characteristics**:
        - Standard humanitarian response needed
        - Regional coordination required
        - Moderate resource allocation
        """)
    
    with col2:
        st.markdown("""
        #### 🟠 Severe Impact
        
        **Characteristics**:
        - Significant humanitarian crisis
        - International assistance needed
        - Substantial resource mobilization
        
        #### 🔴 Critical Impact
        
        **Characteristics**:
        - Major humanitarian emergency
        - Large-scale international response
        - Maximum resource deployment
        """)
    
    st.markdown("""
    ---
    
    ### 🔧 Features Used (No Data Leakage!)
    
    The model uses these features to make predictions:
    """)
    
    features_df = pd.DataFrame({
        'Feature': [
            'Conflict Type',
            'Region',
            'Year',
            'Battle Deaths (Low)',
            'Battle Deaths (High)',
            'Death Range',
            'Uncertainty Ratio',
            'Log(Death Estimate)',
            'Decade',
            'Years Since 1989'
        ],
        'Description': [
            'Type of armed conflict (e.g., interstate, intrastate)',
            'Geographic region of the conflict',
            'Year the conflict occurred',
            'Lower bound casualty estimate',
            'Upper bound casualty estimate',
            'Difference between high and low estimates',
            'Measure of estimate uncertainty',
            'Log-transformed average death estimate',
            'Decade categorization',
            'Temporal distance from dataset start'
        ],
        'Type': [
            'Categorical',
            'Categorical',
            'Numerical',
            'Numerical',
            'Numerical',
            'Derived',
            'Derived',
            'Derived',
            'Derived',
            'Derived'
        ]
    })
    
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    st.warning("""
    **⚠️ Important**: The model does NOT use exact death counts (`bd_best`) as a feature to prevent data leakage. 
    Instead, it uses the low/high estimates and derived features to learn patterns that indicate humanitarian impact.
    """)
    
    st.markdown("""
    ---
    
    ### 🎓 How It Learns
    
    **Training Process**:
    1. **Data Collection**: Historical conflict data from UCDP (1989-present)
    2. **Target Creation**: Humanitarian impact calculated from multiple factors
    3. **Feature Engineering**: Create meaningful derived features
    4. **Model Training**: Machine learning algorithms find patterns
    5. **Validation**: Cross-validation ensures generalization
    6. **Deployment**: Best model selected and deployed
    
    **Why Not Simple Thresholds?**
    - Death counts alone don't capture full humanitarian impact
    - Regional context matters (same deaths = different impacts in different regions)
    - Conflict type affects humanitarian needs
    - Uncertainty in estimates indicates risk
    - Temporal patterns reveal changing conflict dynamics
    
    ---
    
    ### 🎯 Real-World Applications
    
    **1. Humanitarian Organizations**
    - Prioritize resource allocation
    - Plan deployment of field teams
    - Estimate supply needs
    
    **2. International Organizations**
    - Inform funding decisions
    - Coordinate multi-agency responses
    - Track global humanitarian needs
    
    **3. Policy Makers**
    - Early warning for escalation
    - Strategic resource planning
    - Crisis preparedness
    
    **4. Researchers**
    - Analyze humanitarian impact patterns
    - Study conflict-impact relationships
    - Validate intervention effectiveness
    """)

# ============================================================================
# TAB 3: MODEL INSIGHTS
# ============================================================================
with tab3:
    st.header("📈 Model Performance & Insights")
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    with col2:
        st.metric("F1-Score", f"{metadata['f1_score']:.2%}")
    with col3:
        st.metric("CV Mean", f"{metadata['cv_mean_score']:.2%}")
    with col4:
        st.metric("CV Std Dev", f"{metadata['cv_std_score']:.4f}")
    
    st.markdown("---")
    
    # Model information
    st.subheader("🔍 Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Model Information:**
        - **Algorithm**: {metadata['model_name']}
        - **Training Date**: {metadata['train_date']}
        - **Features**: {len(metadata['feature_names'])}
        - **Target Classes**: {len(metadata['target_classes'])}
        - **Uses Scaling**: {'Yes' if metadata.get('uses_scaling', False) else 'No'}
        """)
    
    with col2:
        st.markdown(f"""
        **Target Classes:**
        """)
        for cls in metadata['target_classes']:
            if "Critical" in cls:
                st.markdown(f"- 🔴 {cls}")
            elif "Severe" in cls:
                st.markdown(f"- 🟠 {cls}")
            elif "Moderate" in cls:
                st.markdown(f"- 🟡 {cls}")
            else:
                st.markdown(f"- 🟢 {cls}")
    
    st.markdown("---")
    
    st.subheader("📊 Features Used in Model")
    
    features_list = metadata['feature_names']
    
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, feature in enumerate(features_list):
        with cols[idx % num_cols]:
            st.markdown(f"✓ `{feature}`")
    
    st.markdown("---")
    
    st.subheader("💡 Model Interpretation Guide")
    
    st.markdown("""
    **How to Interpret Predictions:**
    
    1. **High Confidence (>80%)**: 
       - Model strongly agrees this impact level fits the pattern
       - Historical conflicts with similar characteristics had this impact
       - High reliability for planning
    
    2. **Moderate Confidence (60-80%)**: 
       - Model sees mixed signals across features
       - Conflict has characteristics of multiple impact levels
       - Good for initial assessment, verify with experts
    
    3. **Low Confidence (<60%)**: 
       - Unusual combination of features
       - May not match historical patterns well
       - Use with caution, seek additional information
    
    **Best Practices:**
    - ✅ Use predictions as decision support, not replacement
    - ✅ Combine with expert knowledge and ground reports
    - ✅ Consider confidence levels when making critical decisions
    - ✅ Update assessments as new information becomes available
    - ✅ Be aware of model limitations and context
    
    **Limitations:**
    - Model trained on historical data (patterns may change)
    - Cannot account for real-time political developments
    - Regional contexts constantly evolving
    - Uncertainty estimates are approximations
    - Should not be sole basis for life-critical decisions
    """)
    
    st.markdown("---")
    
    st.subheader("🔬 Why This Approach?")
    
    st.success("""
    **Multi-Factor Assessment Prevents Data Leakage:**
    
    Traditional threshold-based classification (e.g., >1000 deaths = War) would achieve 100% accuracy 
    but wouldn't be machine learning - just memorizing rules.
    
    This model:
    - ✓ Learns complex patterns across multiple factors
    - ✓ Doesn't use exact death counts as features
    - ✓ Considers regional and temporal context
    - ✓ Achieves realistic accuracy (70-85%)
    - ✓ Provides actionable humanitarian impact assessment
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Humanitarian Impact Prediction System</strong></p>
    <p>Machine Learning for Developers (CAI2C08) | Temasek Polytechnic</p>
    <p>Multi-Factor ML Approach | No Data Leakage</p>
    <p>⚠️ Educational Tool - For Research Purposes Only</p>
</div>
""", unsafe_allow_html=True)