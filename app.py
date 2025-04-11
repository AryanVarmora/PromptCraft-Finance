# app.py - Streamlit version of PromptCraft-Finance
import os
import json
import torch
import time
from datetime import datetime

# Set environment variables to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create necessary directories
def ensure_project_directories():
    directories = ['prompts', 'saved_responses', 'comparison_results', 'history']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Run this first
ensure_project_directories()

# Try importing required libraries
try:
    import streamlit as st
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nTry installing dependencies with:")
    print("pip install streamlit transformers torch")
    exit(1)

# Define available models with details
model_choices = {
    "Phi-2 (microsoft/phi-2)": {
        "id": "microsoft/phi-2",
        "size": "2.7B parameters"
    },
    "Mistral-7B-Instruct (mistralai)": {
        "id": "mistralai/Mistral-7B-Instruct-v0.1",
        "size": "7B parameters"
    },
    "Falcon RW 1B (tiiuae)": {
        "id": "tiiuae/falcon-rw-1b",
        "size": "1B parameters"
    }
}

# Expanded financial prompt templates
prompts = {
    "budgeting": """Create a monthly budget plan for a person with the following details:
- Income: $5,000/month
- Current expenses: $1,500 rent, $400 car payment, $600 groceries
- Financial goals: Save for down payment on house within 2 years
- Debt: $15,000 in student loans with 4.5% interest rate

Provide a detailed breakdown of recommended spending and saving categories.""",
    
    "investor_profile": """Based on the following investor profile, recommend an appropriate asset allocation:
- Age: 35
- Risk tolerance: Moderate
- Time horizon: 25 years until retirement
- Current investments: $50,000 in 401(k)
- Monthly investment capability: $1,000
- Financial goals: Comfortable retirement and college fund for 2 children

Explain your reasoning for the suggested allocation.""",
    
    "summary": """Summarize the following financial report in simple terms:
Q1 2024 showed a 12% increase in revenue compared to the previous quarter, reaching $2.3M. Gross margin improved from 55% to 58% due to operational efficiencies. However, customer acquisition cost rose by 7%, and the churn rate increased slightly from 2.1% to 2.4%. The company's cash runway is currently 18 months, with $5.4M in available capital. The board has approved an expansion into European markets, projected to begin in Q3.""",

    "debt_repayment": """Create a debt repayment strategy for the following situation:
- Credit card A: $8,000 balance at 18.99% APR
- Credit card B: $5,500 balance at 22.49% APR
- Auto loan: $12,000 remaining at 5.25% APR
- Student loan: $20,000 at 4.5% APR
- Monthly income after essential expenses: $1,200

Compare debt snowball vs. avalanche methods and recommend the best approach.""",

    "real_estate": """Analyze this real estate investment opportunity:
- Purchase price: $350,000
- Down payment: 20% ($70,000)
- Mortgage: 30-year fixed at 5.5% interest
- Estimated monthly rent: $2,100
- Property tax: $3,600/year
- Insurance: $1,200/year
- Estimated maintenance: 1% of property value annually
- HOA fees: $250/month
- Expected property appreciation: 3% annually

Calculate ROI, cash flow, and cap rate. Is this a good investment?""",

    "stock_analysis": """Evaluate this stock based on the following metrics:
- P/E ratio: 18.5
- Price-to-sales: 2.3
- Debt-to-equity: 0.45
- Return on equity: 15.2%
- Current ratio: 1.8
- Five-year revenue growth: 12% annually
- Industry average P/E: 22
- Dividend yield: 2.8%

Would you consider this stock undervalued, fairly valued, or overvalued? Explain your reasoning.""",

    "tax_planning": """Recommend tax optimization strategies for:
- Married couple filing jointly
- Combined income: $180,000
- Two children (ages 8 and 10)
- Mortgage interest: $12,000/year
- 401(k) contributions: Currently $10,000/year
- State income tax: $8,000/year
- Charitable donations: $3,000/year
- Self-employed business income: $45,000 (included in combined income)

What strategies could help reduce their tax liability while building wealth?"""
}

# Create prompt template files
def create_prompt_files():
    for name, content in prompts.items():
        with open(f'prompts/{name}.txt', 'w') as file:
            file.write(content)

# Load prompt templates
def load_prompt(prompt_name):
    try:
        with open(f'prompts/{prompt_name}.txt', 'r') as file:
            return file.read()
    except FileNotFoundError:
        return prompts.get(prompt_name, "")

# Load model pipeline with better error handling
def get_model_pipeline(model_id, max_tokens):
    # Show a message that model is loading
    with st.spinner(f"Loading model {model_id}... This may take a while on first run."):
        try:
            # Use CPU for compatibility
            generator = pipeline(
                "text-generation", 
                model=model_id, 
                max_new_tokens=max_tokens,
                torch_dtype=torch.float32,
            )
            return generator
        except Exception as e:
            st.error(f"Error loading model {model_id}: {str(e)}")
            raise Exception(f"Error loading model {model_id}: {str(e)}")

# Generate response from a single model
def generate_single_response(prompt_text, model_name, max_tokens=300, temperature=0.7):
    if not prompt_text.strip():
        return "⚠️ Error: Please enter a prompt."
    
    try:
        model_id = model_choices[model_name]["id"]
        generator = get_model_pipeline(model_id, max_tokens)
        
        with st.spinner("Generating response..."):
            output = generator(
                prompt_text, 
                do_sample=True, 
                temperature=temperature
            )[0]["generated_text"]
        
        # Process output to remove the input prompt if it's repeated
        if output.startswith(prompt_text):
            output = output[len(prompt_text):].strip()
            if not output:
                output = "Model didn't generate new content beyond the prompt."
        
        # Add to history
        save_to_history(prompt_text, output, model_name)
        
        return output
    except Exception as e:
        return f"⚠️ Error generating text: {e}"

# Generate responses from multiple models for comparison
def generate_comparison(prompt_text, selected_models, max_tokens=300, temperature=0.7):
    if not prompt_text.strip():
        return {model: "⚠️ Error: Please enter a prompt." for model in selected_models}
    
    results = {}
    progress_bar = st.progress(0)
    
    for i, model_name in enumerate(selected_models):
        progress_text = st.empty()
        progress_text.text(f"Processing model {i+1}/{len(selected_models)}: {model_name}")
        
        try:
            model_id = model_choices[model_name]["id"]
            generator = get_model_pipeline(model_id, max_tokens)
            
            output = generator(
                prompt_text, 
                do_sample=True, 
                temperature=temperature
            )[0]["generated_text"]
            
            # Process output to remove the input prompt if it's repeated
            if output.startswith(prompt_text):
                output = output[len(prompt_text):].strip()
                if not output:
                    output = "Model didn't generate new content beyond the prompt."
            
            results[model_name] = output
        except Exception as e:
            results[model_name] = f"⚠️ Error with {model_name}: {e}"
        
        # Update progress
        progress = (i + 1) / len(selected_models)
        progress_bar.progress(progress)
    
    # Save comparison result
    save_comparison(prompt_text, results)
    progress_bar.empty()
    
    return results

# Save single response to history
def save_to_history(prompt, response, model_name):
    history_file = 'history/prompt_history.json'
    
    # Create or load history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
            except:
                history = []
    else:
        history = []
    
    # Add new entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "prompt": prompt,
        "response": response
    }
    
    # Add to beginning of list (most recent first)
    history.insert(0, entry)
    
    # Keep only the 50 most recent entries
    if len(history) > 50:
        history = history[:50]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    return "Response saved to history"

# Load prompt history
def load_history():
    history_file = 'history/prompt_history.json'
    if not os.path.exists(history_file):
        return []
    
    with open(history_file, 'r') as f:
        try:
            return json.load(f)
        except:
            return []

# Save detailed comparison results
def save_comparison(prompt, results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_results/comparison_{timestamp}.json"
    
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "results": results
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    return f"Comparison saved to {filename}"

# Export single response as markdown
def export_response(prompt, response, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_responses/response_{timestamp}.md"
    
    with open(filename, 'w') as file:
        file.write(f"# PromptCraft-Finance Response\n\n")
        file.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(f"**Model:** {model_name}\n\n")
        file.write("## Prompt\n\n```\n")
        file.write(f"{prompt}\n")
        file.write("```\n\n")
        file.write("## Response\n\n")
        file.write(f"{response}\n")
    
    return filename

# Create requirements.txt file
def create_requirements_file():
    requirements = [
        "streamlit>=1.26.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "accelerate>=0.20.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))

# Main Streamlit app
def main():
    # Create files
    create_prompt_files()
    create_requirements_file()
    
    # Set page config
    st.set_page_config(
        page_title="PromptCraft-Finance",
        page_icon="💸",
        layout="wide"
    )
    
    # Header
    st.title("💸 PromptCraft-Finance")
    st.write("Experiment with prompt engineering in financial contexts using open-source LLMs.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Model", 
        "Model Comparison", 
        "History", 
        "Help & Setup"
    ])
    
    # Single Model Tab
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            prompt_selector = st.selectbox(
                "Choose a prompt template",
                options=list(prompts.keys())
            )
            
            model_selector = st.selectbox(
                "Select an LLM Model",
                options=list(model_choices.keys())
            )
            
            with st.expander("Advanced Settings"):
                max_tokens = st.slider(
                    "Maximum Tokens", 
                    min_value=50, 
                    max_value=500, 
                    value=300, 
                    step=50
                )
                
                temperature = st.slider(
                    "Temperature (Creativity)", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.7, 
                    step=0.1
                )
        
        # Load prompt when template selected
        if "current_prompt" not in st.session_state or st.session_state.prompt_selector != prompt_selector:
            st.session_state.current_prompt = load_prompt(prompt_selector)
            st.session_state.prompt_selector = prompt_selector
        
        prompt_text = st.text_area(
            "Prompt", 
            value=st.session_state.current_prompt,
            height=200
        )
        
        col1, col2, col3 = st.columns(3)
        
        generate_btn = col1.button("Generate Response", type="primary")
        clear_btn = col2.button("Clear")
        export_btn = col3.button("Export as Markdown")
        
        # Output area
        output_container = st.container()
        
        # Handle generate button
        if generate_btn:
            response = generate_single_response(
                prompt_text, 
                model_selector, 
                max_tokens, 
                temperature
            )
            st.session_state.current_response = response
            st.session_state.current_model = model_selector
        
        # Handle clear button
        if clear_btn:
            st.session_state.current_prompt = ""
            st.session_state.current_response = ""
        
        # Handle export button
        if export_btn and "current_response" in st.session_state:
            if st.session_state.current_response:
                filename = export_response(
                    prompt_text, 
                    st.session_state.current_response, 
                    st.session_state.current_model
                )
                st.success(f"Response exported to {filename}")
        
        # Display output
        with output_container:
            if "current_response" in st.session_state:
                st.text_area(
                    "Model Output", 
                    value=st.session_state.current_response, 
                    height=300
                )
        
        # Example prompts
        st.write("### Example Prompts")
        example_prompts = [
            "Create a retirement savings plan for a 40-year-old with $100,000 saved and 25 years until retirement.",
            "Compare and contrast index funds vs. actively managed funds for a novice investor.",
            "Explain the concept of dollar-cost averaging and when it's most beneficial."
        ]
        
        for i, example in enumerate(example_prompts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.current_prompt = example
                st.experimental_rerun()
    
    # Model Comparison Tab
    with tab2:
        comparison_prompt = st.text_area(
            "Comparison Prompt", 
            height=200
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_selection = st.multiselect(
                "Select Models to Compare",
                options=list(model_choices.keys()),
                default=[list(model_choices.keys())[0], list(model_choices.keys())[1]]
            )
        
        with col2:
            compare_tokens = st.slider(
                "Maximum Tokens for Comparison", 
                min_value=50, 
                max_value=500, 
                value=300, 
                step=50
            )
            
            compare_temp = st.slider(
                "Temperature for Comparison", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.7, 
                step=0.1
            )
        
        compare_btn = st.button("Compare Models", type="primary")
        
        if compare_btn and comparison_prompt and model_selection:
            results = generate_comparison(
                comparison_prompt, 
                model_selection, 
                compare_tokens, 
                compare_temp
            )
            
            st.session_state.comparison_results = results
        
        # Display comparison results
        if "comparison_results" in st.session_state:
            st.write("### Comparison Results")
            
            # Create columns for results
            cols = st.columns(len(st.session_state.comparison_results))
            
            for i, (model, result) in enumerate(st.session_state.comparison_results.items()):
                with cols[i]:
                    st.subheader(model)
                    st.text_area(
                        "", 
                        value=result, 
                        height=400,
                        key=f"compare_result_{i}"
                    )
        
        # Example comparison prompts
        st.write("### Example Comparison Prompts")
        example_compare_prompts = [
            "What would be a suitable asset allocation for a 30-year-old with moderate risk tolerance?",
            "Explain the difference between traditional and Roth IRAs in simple terms.",
            "How should someone prioritize paying off multiple debts with different interest rates?"
        ]
        
        for i, example in enumerate(example_compare_prompts):
            if st.button(f"Compare Example {i+1}", key=f"compare_example_{i}"):
                st.session_state.comparison_prompt = example
                st.experimental_rerun()
    
    # History Tab
    with tab3:
        st.write("### Prompt History")
        
        if st.button("Refresh History"):
            st.session_state.history_data = load_history()
        
        # Load history initially if not in session state
        if "history_data" not in st.session_state:
            st.session_state.history_data = load_history()
        
        # Display history
        if st.session_state.history_data:
            history_items = [f"{h['timestamp']} - {h['model']}" for h in st.session_state.history_data]
            
            selected_history = st.selectbox(
                "Select from history",
                options=history_items,
                index=0
            )
            
            if selected_history:
                # Find the selected history item
                selected_index = history_items.index(selected_history)
                history_entry = st.session_state.history_data[selected_index]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_area(
                        "Historical Prompt", 
                        value=history_entry["prompt"], 
                        height=200
                    )
                
                with col2:
                    st.text_area(
                        "Historical Response", 
                        value=history_entry["response"], 
                        height=200
                    )
                
                # Option to reuse this prompt
                if st.button("Use this prompt", key="use_history_prompt"):
                    st.session_state.current_prompt = history_entry["prompt"]
                    st.session_state.tab_selection = "Single Model"
                    st.experimental_rerun()
        else:
            st.info("No history found. Generate some responses first.")
    
    # Help Tab
    with tab4:
        st.markdown("""
        # Setup Guide
        
        ## Installation
        
        1. Clone the repository:
        ```
        git clone https://github.com/AryanVarmora/PromptCraft-Finance.git
        cd PromptCraft-Finance
        ```
        
        2. Install dependencies:
        ```
        pip install -r requirements.txt
        ```
        
        3. Run the application:
        ```
        streamlit run app.py
        ```
        
        ## First Run Tips
        
        - When you first select a model, it will download automatically (may take time)
        - Models are saved locally for future use
        - For large models like Mistral-7B, ensure 16GB+ RAM (or use smaller models)
        
        ## Troubleshooting
        
        If you encounter errors:
        
        - Try a smaller model first like Falcon RW 1B
        - Reduce the maximum token count
        - Check your internet connection for model downloads
        - Ensure you have sufficient disk space
        
        ## About PromptCraft-Finance
        
        This app lets you experiment with prompt engineering for financial use cases. Compare how different open-source LLMs respond to the same prompt and refine your prompts for better results.
        """)

if __name__ == "__main__":
    main()