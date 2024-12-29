import streamlit as st
from dotenv import load_dotenv
import modules.embeddings as emb
import modules.llm as llm
from modules.output_parser import parse_text
import json
from htmlTemplates import css, bot_template, user_template

load_dotenv()


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

def generate_json():
    if len(st.session_state.chat_history) > 0:
        with st.spinner("Procesando JSON..."):
            parsed_output = parse_text(st.session_state.chat_history[-1].content)
            st.session_state["parsed_output"] = json.dumps(parsed_output, indent=4)
            st.success("JSON Generado!")

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
    if "namespaces" not in st.session_state:
        st.session_state.namespaces = list(emb.retrieveNamespace())
    if "namespaceSelected" not in st.session_state:
        st.session_state.namespaceSelected = None
    if "sidebar_enable" not in st.session_state:
        st.session_state.sidebar_enable = True
    if "dropdown_enabled" not in st.session_state:
        st.session_state.dropdown_enabled = True
    if "parsed_output" not in st.session_state:
        st.session_state.parsed_output = None
    if "namespace_input" not in st.session_state:
        st.session_state.namespace_input = False
    if "namespace_add_content_button" not in st.session_state:
        st.session_state.namespace_add_content_button = True
        
    
    st.header("Lerni - Generador de Lecciones", anchor=False)
    
    st.sidebar.title("Configuración")
    
    st.sidebar.write("Selector de Contenedor:")
    namespace_selected = st.sidebar.selectbox(" ", st.session_state.namespaces, help="Seleccionar el contenedor que se desea utilizar para generar las lecciones.", disabled=not st.session_state.sidebar_enable)
    
    if namespace_selected == 'default':
        namespace_selected = ""
    
    if namespace_selected != st.session_state.namespaceSelected:
        st.session_state.namespaceSelected = namespace_selected
        st.rerun()
    
    # Crear Namespace
    st.sidebar.divider()
    st.sidebar.write("Agregar Contenedor:")
    if st.session_state.namespace_add_content_button:
        if st.sidebar.button("Habilitar Input", key="modal", use_container_width=True, disabled=not st.session_state.sidebar_enable):
            st.session_state.namespace_input = True
            st.session_state.namespace_add_content_button = False
            st.rerun()
    
    
    if st.session_state.namespace_input:
        st.sidebar.info("Agrega un contenedor nuevo al selector de contenido, es necesario subir un archivo despues que selecciones el contenedor nuevo.")
        new_namespace = st.sidebar.text_input("Nombre del Contenedor Nuevo:", key="namespace", disabled=not st.session_state.sidebar_enable, placeholder="Ej. Patroñes de Diseño")
        if st.sidebar.button("Guardar", key="save_namespace", disabled=not st.session_state.sidebar_enable, use_container_width=True):
            if new_namespace not in st.session_state.namespaces:
                st.session_state.namespaces.append(new_namespace)
                st.session_state.namespace_input = False
                st.session_state.namespace_add_content_button = True
                st.rerun()
    
    if not st.session_state.input_enabled:
        if st.button("Iniciar", type="primary"):
            st.session_state.input_enabled = True
            st.session_state.sidebar_enable = False
            with st.spinner("Procesando..."):
                vectorstore = emb.get_vectorstore(namespace=st.session_state.namespaceSelected)
                st.session_state.conversation = llm.get_conversation_chain(vectorstore)
            st.rerun()
                
    if st.session_state.input_enabled:
        with st.form(key='user_input_form', clear_on_submit=True, border=False):
            user_question = st.text_area(" ", key="user_question", height=125, placeholder="Escribe tus indicaciones aquí.")
            submit_button = st.form_submit_button(label='Enviar')
            
            if submit_button and user_question.strip():
                if user_question.endswith('\n'):
                    user_question = user_question[:-1]
                with st.spinner("Generando leccion..."):
                    handle_userinput(user_question.strip())
                    
    if st.session_state.chat_history:
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    st.sidebar.divider()
    st.sidebar.write("Subir documentos:")
    with st.sidebar:
        files = st.file_uploader("Subir sus documentos y seleccionar 'Procesar'", accept_multiple_files=True, type=['pdf', 'mp3', 'mp4'], help='Procesamiento de documentos nuevos al contenido seleccionado',disabled=not st.session_state.sidebar_enable)
        if st.button("Procesar", disabled=not st.session_state.sidebar_enable, use_container_width=True):
            with st.spinner("Procesando..."):
                for file in files:
                    if "video" in file.type:
                        print("video")
                        emb.get_video_transcription(file, st.session_state.namespaceSelected)
                    if "audio" in file.type:
                        print("audio")
                        emb.get_audio_transcription(file.getvalue(), file.name, st.session_state.namespaceSelected)
                    if "pdf" in file.type:
                        print("pdf")
                        emb.get_pdf_text(file, st.session_state.namespaceSelected)
            st.success("Subida de archivo completa!")
              
        # Generate JSON Button
        if len(st.session_state.chat_history) > 0:
            st.sidebar.divider()
            st.sidebar.write("Generador de JSON:")
            if st.button("Generar JSON", use_container_width=True):
                generate_json()

        # Download JSON Button (only available if JSON is generated)
        if st.session_state.parsed_output:
            st.download_button(
                label="Descargar JSON",
                data=st.session_state.parsed_output,  # Access the generated JSON from session state
                file_name="output.json",
                mime="application/json",
                key=None,
                help=None,
                type="secondary",
                disabled=False,
                use_container_width=True
            )



if __name__ == "__main__":
    main()