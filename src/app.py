import streamlit as st
from dotenv import load_dotenv
import modules.embeddings as emb
import modules.llm as llm
from htmlTemplates import css, bot_template, user_template

load_dotenv()
    

def handle_userinput(user_question):
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

def main():

    st.set_page_config(page_title="Lerni - Generador de lecciones", 
                       page_icon=":robot_face:" )
    
    st.write(css, unsafe_allow_html=True)
    
    # Inicializar la session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "input_enabled" not in st.session_state:
        st.session_state.input_enabled = False
    
    st.header("Generador de Lecciones - Austral :robot_face:")

    if not st.session_state.input_enabled:
        if st.button("Iniciar", type="primary"):
            st.session_state.input_enabled = True
            with st.spinner("Procesando"):
                vectorstore = emb.get_vectorstore()
                st.session_state.conversation = llm.get_conversation_chain(vectorstore)
            st.rerun()
                
    if st.session_state.input_enabled:
        with st.form(key='user_input_form', clear_on_submit=True, border=False):
            user_question = st.text_area(" ", key="user_question", height=125, placeholder="Escribe tus indicaciones")
            submit_button = st.form_submit_button(label='Enviar')
            
            if submit_button and user_question.strip():
                if user_question.endswith('\n'):
                    user_question = user_question[:-1]
                with st.spinner("Generando leccion..."):
                    handle_userinput(user_question.strip())
                    
    if st.session_state.chat_history:
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__ == "__main__":
    main()