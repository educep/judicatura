"""
Ejemplos específicos para la presentación judicial
Created by Analitika at 07/09/2024
contact@analitika.fr
"""

# Ejemplos de consultas para demostración
JUDICIAL_QUERIES = {
    "derecho_laboral": [
        "casos de violación de derecho al debido proceso en juicios laborales",
        "precedentes sobre despido injustificado en Ecuador",
        "jurisprudencia sobre horas extras no pagadas"
    ],
    "derecho_civil": [
        "precedentes sobre indemnización por daño moral en accidentes de tránsito",
        "jurisprudencia sobre incumplimiento de contratos de compraventa",
        "casos de responsabilidad civil extracontractual"
    ],
    "derecho_penal": [
        "jurisprudencia sobre robo agravado y penas aplicables",
        "precedentes sobre tráfico de sustancias estupefacientes",
        "casos de violencia intrafamiliar y medidas de protección"
    ],
    "derecho_constitucional": [
        "acciones de protección por violación de derechos fundamentales",
        "jurisprudencia sobre derecho al debido proceso",
        "casos de acceso a la justicia y barreras económicas"
    ],
    "pensiones_alimenticias": [
        "cálculo de pensión alimenticia según capacidad económica",
        "jurisprudencia sobre incremento de pensiones alimenticias",
        "casos de incumplimiento de obligaciones alimentarias"
    ]
}

# Ejemplos de respuestas del sistema
SAMPLE_RESPONSES = {
    "derecho_laboral": {
        "query": "casos de violación de derecho al debido proceso en juicios laborales",
        "results": [
            {
                "sentencia": "123-2024-LA",
                "extracto": "El derecho al debido proceso laboral incluye garantías de audiencia y defensa, conforme al Art. 75 de la Constitución.",
                "relevancia": 0.95
            },
            {
                "sentencia": "089-2024-LA", 
                "extracto": "La violación del debido proceso en despidos genera nulidad del acto y derecho a reinstalación.",
                "relevancia": 0.92
            },
            {
                "sentencia": "156-2024-LA",
                "extracto": "El empleador debe notificar previamente las causas del despido para garantizar el debido proceso.",
                "relevancia": 0.89
            }
        ]
    },
    "derecho_civil": {
        "query": "precedentes sobre indemnización por daño moral en accidentes de tránsito",
        "results": [
            {
                "sentencia": "234-2024-CV",
                "extracto": "La indemnización por daño moral se calcula considerando el impacto psicológico demostrado.",
                "relevancia": 0.96
            },
            {
                "sentencia": "189-2024-CV",
                "extracto": "Se debe considerar la capacidad económica del responsable para fijar la indemnización.",
                "relevancia": 0.91
            },
            {
                "sentencia": "301-2024-CV",
                "extracto": "La gravedad del accidente es factor determinante en el cálculo del daño moral.",
                "relevancia": 0.88
            }
        ]
    }
}

# Ejemplos de inconsistencias detectadas
CONSISTENCY_EXAMPLES = [
    {
        "tema": "Cálculo de pensión alimenticia",
        "juzgado_a": {
            "nombre": "Juzgado A",
            "criterio": "30% del ingreso del obligado",
            "sentencia": "123-2024-FA"
        },
        "juzgado_b": {
            "nombre": "Juzgado B", 
            "criterio": "25% del ingreso del obligado",
            "sentencia": "456-2024-FA"
        }
    },
    {
        "tema": "Indemnización por daño moral",
        "juzgado_a": {
            "nombre": "Juzgado A",
            "criterio": "Mínimo 3 salarios básicos",
            "sentencia": "789-2024-CV"
        },
        "juzgado_b": {
            "nombre": "Juzgado B",
            "criterio": "Mínimo 5 salarios básicos", 
            "sentencia": "012-2024-CV"
        }
    }
]

# Consultas ciudadanas y respuestas
CITIZEN_QUERIES = {
    "pension_alimenticia": {
        "query": "¿Qué dice la ley sobre pensión alimenticia?",
        "response": "La pensión alimenticia es un derecho fundamental que garantiza el sustento de hijos menores. Se calcula según la capacidad económica del obligado y las necesidades del beneficiario, conforme al Código Civil ecuatoriano.",
        "informacion_adicional": [
            "Procedimiento para solicitar pensión",
            "Documentos requeridos",
            "Plazos legales aplicables",
            "Recursos disponibles"
        ]
    },
    "accidente_transito": {
        "query": "¿Qué hacer en caso de accidente de tránsito?",
        "response": "En caso de accidente de tránsito, debe reportar inmediatamente a las autoridades, solicitar asistencia médica si es necesario, y recopilar evidencia del siniestro para futuras reclamaciones.",
        "informacion_adicional": [
            "Pasos inmediatos a seguir",
            "Documentos a recopilar",
            "Plazos para reclamaciones",
            "Derechos y obligaciones"
        ]
    }
}

def get_judicial_examples(category="all"):
    """
    Obtiene ejemplos judiciales por categoría
    """
    if category == "all":
        return JUDICIAL_QUERIES
    return JUDICIAL_QUERIES.get(category, {})

def get_sample_response(category):
    """
    Obtiene respuesta de muestra para una categoría
    """
    return SAMPLE_RESPONSES.get(category, {})

def get_consistency_examples():
    """
    Obtiene ejemplos de inconsistencias judiciales
    """
    return CONSISTENCY_EXAMPLES

def get_citizen_query(topic):
    """
    Obtiene consulta ciudadana y respuesta
    """
    return CITIZEN_QUERIES.get(topic, {})
