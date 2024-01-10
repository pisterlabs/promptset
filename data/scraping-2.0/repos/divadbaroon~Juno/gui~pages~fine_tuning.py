import streamlit as st
from training.begin_gpt_training_session import OpenAIFineTuningJob
from src.utilities.settings.master_settings.master_settings_manager import MasterSettingsManager
from src.customization.profiles.profile_manager import ProfileManager
from src.customization.voices.voice_settings_manager import VoiceSettingsManager
from configuration.manage_secrets import ConfigurationManager

# Check if page_state is already initialized
if 'page_state' not in st.session_state:
    st.session_state.page_state = "default"

def fine_tune_GPT():
    """
    Initiates the fine-tuning process for the GPT model.
    """
    st.subheader("🛠 Start Fine-Tuning Process")
    st.write("Provide a unique name for your fine-tuned GPT model and upload the required training data in `.jsonl` format.")
    
    model_name = st.text_input("Model Name:")
    file = st.file_uploader("Training Data (.jsonl format)", type=["jsonl"])
    
    if st.button("Begin Training"):
        if model_name and file:
            OpenAIFineTuningJob().start_finetuning(model_name=model_name, file=file)
            st.success("🎉 Training started successfully!")
            st.info("📩 An email notification from OpenAI will arrive shortly with your job ID. Once received, you can use the integration section below to add your new model into Juno.")
        else:
            st.warning("⚠️ Ensure you provide both a model name and the training data to proceed.")

def add_model_to_juno():
    """
    Integrates a fine-tuned GPT model into Juno.
    """
    st.subheader("🔗 Integrate Your Fine-Tuned Model with Juno")
    st.write("Enter the job ID provided by OpenAI to integrate your newly trained model with Juno.")
    model_name = st.text_input("Enter A Model Name:")
    model_id = st.text_input("Job ID from OpenAI:")

    if st.button("Integrate Model"):
        
        profile_name = MasterSettingsManager().retrieve_property('profile')
        ProfileManager().save_property('gpt_model', model_name, profile_name)
        ConfigurationManager().add_secret(secret_name=model_name, secret_value=model_id )

        st.success(f"🌟 Model {model_name} has been integrated with Juno!")
        
def overview():
    st.title("🚀 Fine-Tuning GPT")
    st.write("""
    Juno's fine-tuning interface lets you optimize GPT for your specific tasks, 
    enhancing its performance and accuracy. You can train a new model and then 
    integrate it directly into Juno.
    """)

def test_model():
    pass

def fine_tuning_interface():
    
    # Main Interface
    menu_functions = {
        "📘 About": overview,
        '🎓 Train a New Model': fine_tune_GPT,
        '🔗 Integrate Model with Juno': add_model_to_juno,
        '✅ Test your Model': test_model
    }

    options = list(menu_functions.keys())
    activity = st.sidebar.selectbox("Select an Option", options)

    menu_functions[activity]()

