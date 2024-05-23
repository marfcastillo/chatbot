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
            "responses": ["¡Hasta luego! Espero haberte servido de mucha ayuda"]
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
        {   "intent": "modalidades",
            "patterns": ["Hay opciones de estudio online o semipresencial", "Puedo estudiar en linea", "Hay opcion de estudio en linea", "Puedo estudiar online",],
            "responses": ["Puedes optar por la modalidad en linea si no te encuentras en el estado Lara. Para solicitar las clases online, debe dirigirse a la oficina del Edificio N"]
            
        },
        {
            "intent": "horario decanato",
            "patterns": ["Horario del DCYT", "Cual es el horario de atencion?", "Horario de atencion en el decanato", "Sabes cual es el horario de atencion en el decanato?", "Cual es el horario de atencion en el decanato?",  "Horario de atencion en el DCYT", "Horario de oficina en el decanato"],
            "responses": ["El horario de atencion es de lunes a viernes de 8:00 AM a 12:00 PM", "Puedes dirigirte a la oficina de 8:00 AM a 12:00 PM"]
        },
        {
            "intent": "tiempo limite",
            "patterns": ["Tiempo limite para incluir una materia", "Tiempo limite para añadir una materia", "Cual es el tiempo limite para incluir una materia", "Cual es el tiempo limite para añadir una materia", "Limite para incluir una materia", "Limite para añadir una materia"],
            "responses": ["El tiempo limite para incluir una materia es 1 semana luego de las inscripciones", "Solicita la inclusion durante una semana despues de las inscripciones en el edificio N"]	
        },
        {
            "intent": "academico",
            "patterns": ["Como puedo contactar al registro academico", "Donde queda registro academico"],
            "responses": ["Registro academico se encuentra en el edificio N al lado de la entrada principal. Puedes realizar tus solicitudes entre 8:00 AM a 12:00 PM"]	
        },
        {
            "intent": "constancia",
            "patterns": ["Como puedo obtener la constancia de estudio", "Necesito la constancia de estudio. Donde puedo conseguirla","Puedo conseguir la constancia de estudio en linea?", "Donde debo dirigirme para obtener la constancia de estudio"],
            "responses": ["Para obtener la constancia de estudio debes dirigirte al Rectorado para solicitar la constancia. El Decanato se encuentra en la Carrera 19 entre Calles 8 y 9 https://www.youtube.com/", "En el Rectorado del Decanato, el cual se encuentra en la Carrera 19 entre calle 8 y 9"]	
        },
        {
            "intent": "biblioteca",
            "patterns": ["Como obtengo el registro de la biblioteca en el decanato", "Como me registro en la biblioteca", "Como hago uso de los libros en la biblioteca", "Como es el proceso de registro en la biblioteca"],
            "responses": ["Para registrate en la bilioteca, debes llevar la planilla de inscripcion impresa descargable en CumLaude, una foto tipo Carnet y tu Carnet UCLA estudiantil."]	
        },
        {
            "intent": "extracurricular",
            "patterns": ["Dime las actividades extracurriculares del DCYT", "Cuales actividades extracurriculares puedo realizar", "Conoces de alguna actividad extracurricular que pueda hacer", "Actividades extracurriculares en el DCYT"],
            "responses": ["Actualmente solo puedes unirte a las actividades deportivas como el equipo de Basketball y Futbol. Por otro lado, existen grupos organizados como M.O.U.S.E, CUMBRE DCyT y Bobby Fisher DCyT"]	
        },
        {
            "intent": "becas",
            "patterns": ["Como puedo obtener una beca universitaria", "El DCyT tiene programas de becas", "Existen becas universitarias que pueda conseguir", "Como es el programa de becas en el DCyT"],
            "responses": ["Lo siento, por los momentos no se encuentran disponibles las Becas"]	
        },
        {
            "intent": "extraordinarias",
            "patterns": ["Como es el proceso de extraordinarias","Que debo hacer para realizar las extraordinarias", "Como puedo inscribirme en las extraordinarias"],
            "responses": ["El proceso de extraordinarias comienza luego de las inscripciones, para registrarte, deberas dirigirte a la oficina de Registro Academico para presentar la solicitud de la materia de tu eleccion"]	
        },
        {
            "intent": "transporte",
            "patterns": ["Existe transporte disponible en el DCyT","Cuales son las opciones de transporte en el decanato", "El decanato cuenta con transporte"],
            "responses": ["Lo siento, actualmente no existe transporte disponible en el DCyT"]	
        },
        {
            "intent": "inscripcion",
            "patterns": ["Cuando son las proximas inscripciones en el DCYT", "Cuando son las proximas inscripciones", "Conoces la fecha de inscripcion del proximo semestre", "Cuando debo inscribirme para el proximo semestre"],
            "responses": ["Las proximas inscripciones en el DCYT comienzan el 9 de diciembre del 2024"]	
        },
        {
            "intent": "disponibilidad pasantias",
            "patterns": ["El decanato ofrece programas de pasantia", "El decanato ofrece pasantias", "Puedo hacer pasantias en el decanato"],
            "responses": ["Si, el decanato ayuda a los estudiantes del X semestre conseguir las pasantias necesarias segun la carrera. Para ello, debes contactarte con el encargado de pasantias el cual podra ofrecer diferentes empresas que aceptan pasantes"]	
        },
        {
            "intent": "empresas pasantias",
            "patterns": ["Que empresas colaboran con el decanato para las pasantias", "Con que empresas puedo hacer pasantias", "Que empresa facilita pasantias a traves del decanato", "Cuales empresas colaboran con el decanato para las pasantias"],
            "responses": ["No hay especificaciones con respecto a las empresas que aceptan pasantias, todo dependera del tiempo en que realices las pasantias pero no te preocupes, el encargado te puede facilitar esta informacion apenas este disponible para que escojas. Tambien te recomiendo buscar por tu lado las empresas que ofrezcan oportunidades de pasantia"]	
        },
        {
            "intent": "acceder pasantias",
            "patterns": ["Como acceder a las pasantias que ofrece el decanato", "Necesito acceder a las pasantias", "Como puedo acceder a las pasantias ofrecidas por el decanato", "Que debo hacer para acceder a las ofertas de pasantias del decanato"],
            "responses": ["Para acceder a las pasantias, debes comunicarte con los profesores encargados de las pasantias los cuales te guiaran en el proceso de solicitud y aprobacion asi como presentarte las opciones ofrecidas por diferentes empresas"]	
        },
        {
            "intent": "requisitos pasantias",
            "patterns": ["Que requisitos debo cumplir para realizar las pasantias", "Cuales son los requisitos para realizar las pasantias", "Existen requisitos para realizar las pasantias", "Cuales son los requisitos para hacer las pasantias"],
            "responses": ["El requisito principal es aprobar los 9 semestres anteriores, luego de esto, solo debes presentar tus datos personales al encargado de pasantias para crear la carta de solicitud. Facil!"]	
        },
        {
            "intent": "graduar pasantias",
            "patterns": ["Necesito hacer pasantias para graduarme", "Es necesario hacer pasantias para graduarme"],
            "responses": ["Solamente sera obligatorio si es parte del pensum de la carrera. Por ejemplo, Ingenieria en Informatica tiene la opcion de hacer pasantias o presentar un trabajo de grado pero en el caso de Ingenieria en Telematica, es obligatorio realizar pasantias"]	
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
    doc = [token for token in doc if not token.is_stop] #David's input
    main_intent = "default"
    
    #Keywords
    intent_keywords = {
        "saludo": ["hola", "buenos", "días", "estás"],
        "despedida": ["adiós", "adios", "luego", "pronto"],
        "informacion": ["qué", "es", "explica", "definir"],
        "carreras": ["carreras", "ingenieria", "ofrece", "disponibles"],
        "tiempo telematica": ["telematica", "ingenieria", "telematica", "toma", "años", "semestres", "carrera", "dura"],
        "tiempo informatica": ["informatica", "ingenieria", "informatica", "toma", "años", "semestres", "carrera", "dura"],
        "modalidades": ["en linea", "online", "semipresencial", "estudiar", "opcion","opciones", "linea", "presencial"],
        "horario decanato": ["cual","es","horario", "atencion", "oficina", "horario del decanato","horas", "consultas", "atencion del decanato"],
        "tiempo limite": ["limite", "incluir", "añadir", "materias", "tiempo", "solicitar", "materia"],	
        "academico": ["contactar", "academico", "donde", "solicitar","registro academico"],	
        "constancia": ["constancia", "estudio", "obtener", "solicitar", "constancia de","constancia de estudio", "constancia"],
        "biblioteca": ["biblioteca","registro", "libros", "biblioteca registro", "registro de la"],	
        "extracurricular": ["actividades", "extracurricular","extracurriculares", "actividad"],
        "becas": ["becas", "beca", "universitaria", "becas universitarias"],
        "extraordinarias": ["extraordinarias", "extraordinaria"],
        "transporte": ["transporte", "transporte disponible"],
        "inscripcion": ["inscripciones", "inscripcion", "inscribirme"],
        "disponibilidad pasantias": ["disponibilidad", "ofrecer", "programa"],
        "empresas pasantias": ["empresas", "colaboran","practicas"],
        "acceder pasantias": ["aplicar", "acceder", "ofertas", "acceder pasantias"],
        "requisitos pasantias": ["requisitos", "condiciones", "cumplir","requisitos pasantias"],
        "graduar pasantias": ["graduarse", "realizar pasantias", "grado", "necesario", "graduarme","graduar"]
        
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
