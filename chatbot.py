import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import spacy


nltk.download('punkt')

nlp = spacy.load("es_core_news_sm")

#Knowledge base on JSON
knowledge_base = {
    "intents": [
        {
            "intent": "saludo",
            "patterns": ["Hola", "Buenos días", "Hola, cómo estás"],
            "responses": ["¡Hola soy BIA! La asistente virtual del DCYT ¿En qué puedo ayudarte?"]
        },
        {
            "intent": "despedida",
            "patterns": ["Adiós", "Adios" ,"Hasta luego", "Hasta pronto"],
            "responses": ["¡Hasta luego!"]
        },
        {
            "intent": "default",
            "patterns": ["", "No entiendo", "No sé"],
            "responses": ["Lo siento, no puedo entender esa pregunta."]
        },
        {
            "intent": "carreras",
            "patterns": ["Carreras de ingenieria que ofrece el decanato", "Carreras de ingenieria en el DCYT",  "Carreras disponibles en el decanato"],
            "responses": ["En el DCYT puedes cursar 8 materias actualmente: Ingenieria en Telematica, Ingenieria Informatica, Ingeniera en Produccion, Analisis de Sistemas, Matematicas y Fisica"]
        },
        {   "intent": "tiempo telematica",
            "patterns": ["Tiempo de estudio en telematica", "Cuanto toma la carrera de telematica", "Cuanto dura la carrera de telematica"],
            "responses": ["Ingenieria en Telematica tiene un tiempo estimado de 5 años para el curso de 10 semestres", "5 años aproximadamente"]
            
        },
        {   "intent": "tiempo informatica",
            "patterns": ["Tiempo de estudio en informatica", "Cuanto toma la carrera de informatica", "Cuanto dura la carrera de informatica"],
            "responses": ["Ingenieria en Informatica tiene un tiempo estimado de 5 años para el curso de 10 semestres", "5 años aproximadamente"]
            
        },
        {
            "intent": "horario decanato",
            "patterns": ["Horario del DCYT", "Cual es el horario de atencion?", "Horario de atencion en el decanato", "Sabes cual es el horario de atencion en el decanato?", "Cual es el horario de atencion en el decanato?",  "Horario de atencion en el DCYT", "Horario de oficina en el decanato"],
            "responses": ["El horario de atencion es de lunes a viernes de 8:00 AM a 12:00 PM", "Puedes dirigirte a la oficina de 8:00 AM a 12:00 PM"]
        }
    ]
}

def preprocess_input(input_text):
    tokenizer = RegexpTokenizer('\s+', gaps = True)
    tokens = tokenizer.tokenize(input_text)
    return tokens

def classify_intent(input_text):
    if isinstance(input_text, list):
        input_text = ' '.join(input_text)
    
    #Tokenize
    doc = nlp(input_text)
    # Remove stopwords from tokens
    doc = [token for token in doc if not token.is_stop]
    main_intent = "default"
    
    #Keywords
    intent_keywords = {
        "saludo": ["hola", "buenos", "días", "estás"],
        "despedida": ["adiós", "adios", "luego", "pronto"],
        "informacion": ["qué", "es", "explica", "definir"],
        "carreras": ["carreras", "ingenieria", "ofrece", "disponibles"],
        "tiempo telematica": ["telematica", "ingenieria", "telematica", "toma", "años", "semestres", "carrera", "dura"],
        "tiempo informatica": ["informatica", "ingenieria", "informatica", "toma", "años", "semestres", "carrera", "dura"],
        "horario decanato": ["cual","es","horario", "atencion", "oficina", "horario del decanato","horas", "consultas", "atencion del decanato"]
    }

    #intent check
    for intent, keywords in intent_keywords.items():
        if any(token.text.lower() in keywords for token in doc):
            main_intent = intent
            break
    
    return main_intent

def get_response(intent):
    for intent_data in knowledge_base["intents"]:
        if intent_data["intent"] == intent:
            return random.choice(intent_data["responses"])
    
    return "Lo siento, no puedo entender esa pregunta."

def chat_bot():
    print("¡Hola soy BIA! La asistente virtual del DCYT ¿En qué puedo ayudarte?")
    
    while True:
        user_input = input('Usuario: ')
        
        if user_input.lower() == 'quit':
            print("BIA: ¡Hasta luego!")
            break
        
        tokens = preprocess_input(user_input)
        intent = classify_intent(tokens)
        response = get_response(intent)
        print(f'BIA: {response}')

if __name__ == '__main__':
    chat_bot()
