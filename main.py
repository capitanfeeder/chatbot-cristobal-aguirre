import subprocess
import sys
import os
import time
import streamlit as st
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from streamlit_chat import message
from examenes import cristobal_aguirre
from openai import OpenAI

# Configuración del backend
app = FastAPI()

# Obtener la API key de la variable de entorno
deepseek_api_key = st.secrets.get("deepseek_api_key")

# Endpoint para interactuar con el chatbot
@app.post("/chat")
async def chat(request: Request):
    try:
        actual_date = time.strftime("%Y-%m-%d")
        data = await request.json()
        question = data.get("prompt")
        if not question:
            return JSONResponse(content={"error": "No se proporcionó una pregunta"}, status_code=400)
        
        # Instrucción inicial para el chatbot
        instruction = f"""Eres un asistente del 'Instituto Superior de Formación Docente, Continua y Técnica Don 
                          Cristobal de Aguirre' (ISFDCyT por sus siglas) de la ciudad de Clorinda Formosa que 
                          debe responder a las preguntas del usuario: {question} y poder despejar dudas sobre fechas 
                          de exámenes finales.
                          
                          Importante:
                          - La fecha actual es {actual_date}.
                          - Recuerda al usuario al finalizar tu respuesta que las fechas de inscripción para exámenes 
                            finales son entre el 2 y el 4 de Diciembre de 2024.
                          - Solo necesitas saber el nombre de la materia o su abreviacion, el nombre de la carrera o su 
                            abreviacion y el año-curso del alumnos. Ejemplo: 'Dame la fecha de examen final de METODOLOGÍA 
                            de la carrera de PERIODISMO de 2° año'
                          - Si te piden información acerca de fechas de exámenes finales debes sacarlos de este 
                            lugar: {cristobal_aguirre}.
                          - Acá solo hay fechas de exámenes finales. Siempre busca primero la materia que te piden 
                            y luego revisa carrera y año.
                            """
        
        # Generar la respuesta con el modelo de DeepSeek
        client = OpenAI(
            api_key=deepseek_api_key,  
            base_url="https://api.deepseek.com"  
        )
        
        completion = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": instruction}
            ],
            temperature=0.5,
            top_p=1,
            max_tokens=512,
            stream=False
        )
        
        response_text = completion.choices[0].message.content
        return JSONResponse(content={"response": response_text}, status_code=200)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return JSONResponse(content={"error": "Error interno del servidor"}, status_code=500)

# Configuración del frontend
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Endpoint del chatbot
API_URL = "http://localhost:8000/chat"

# Función para enviar la pregunta al chatbot y obtener la respuesta
def get_response(prompt):
    response = requests.post(API_URL, json={"prompt": prompt})
    if response.status_code == 200:
        json_response = response.json()
        return json_response.get("response", "No se pudo obtener una respuesta.")
    else:
        return "Error al comunicarse con el servidor."

# Inicializar el estado de la conversación
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Título del chatbot
st.title("Chatbot I.S.F.D.C.y T. 'Don Cristóbal de Aguirre' Clorinda")

# Mostrar la conversación
for i in range(len(st.session_state['generated'])):
    message(st.session_state['past'][i], is_user=True, key=f"past_{i}")
    message(st.session_state['generated'][i], key=f"generated_{i}")

# Área de entrada de texto
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    st.session_state.past.append(prompt)
    message(prompt, is_user=True, key=f"past_{len(st.session_state.past) - 1}")

    with st.spinner("Generando respuesta..."):
        response = get_response(prompt)
        st.session_state.generated.append(response)
        message(response, key=f"generated_{len(st.session_state.generated) - 1}")

# Función principal para iniciar la aplicación
def main():
    print("Iniciando la aplicación...")
    
    # Iniciar el servidor FastAPI en un proceso separado
    backend_process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000', '--reload'])
    
    # Esperar un poco para asegurarse de que el backend esté listo
    time.sleep(5)
    
    # Iniciar la aplicación Streamlit
    frontend_process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'main.py', '--server.port', '8501'])
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo la aplicación...")
    
    backend_process.terminate()
    frontend_process.terminate()

if __name__ == "__main__":
    main()
