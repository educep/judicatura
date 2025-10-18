"""
Created by Analitika at 07/09/2024
contact@analitika.fr
"""
import os

from dotenv import load_dotenv

load_dotenv()
aws_path = os.getenv("AWS_PATH")

favicon = f"{aws_path}/DS_Imagen_de_marca/logos/favicondsazul.png"
sample_markdown = fr"""
<section style="background-color: #098DFA; color: white; text-align: center;">
    <div style="margin-top: 15vh;">
        <h1 style="font-size: 1.5em; font-weight: bold;">De horas a minutos</h1>
        <h1 style="font-size: 1.5em; font-weight: bold;">Como la IA revoluciona la</h1>
        <h2 style="font-size: 1.0em; font-weight: bold;">Investigación Jurídica</h2>
        <h3 style="font-size: 1.2em;">Judicatura - Quito</h3>
        <p style="font-size: 1.1em; margin-top: 1.5em;"> Octubre 2025</p>
    </div>
</section>
---
<!-- .slide: data-background-image="{aws_path}/DS_Imagen_de_marca/SLIDES/Presentacion_V1_01.png" data-background-size="115% 100%" data-background-position="center" -->

<section style="position: relative; top: 20%; left: 10%; display: flex; justify-content: space-between; align-items: center; width: 90%;">
    <!-- Left content: Text -->
    <div style="position: absolute; top: 30%; left: 10%; text-align: left; color: white; max-width: 50%;">
        <h2 style="color: white;">¿Quién soy?</h2>
        <strong>Eduardo Cepeda, Ph.D.</strong> <br>
        <em>CEO & Founder</em> <br>
        📞 +33 (0)6 50 90 01 49 <br>
        ✉️ <a href="mailto:eduardo@datoscout.ec" style="color: white;">eduardo@datoscout.ec</a> <br>
        🌐 <a target="_blank" href="http://www.datoscout.ec" style="color: white;">www.datoscout.ec</a>
    </div>
    <!-- Image positioned on the right side using CSS -->
    <div style="position: absolute; top: 20%; right: 10%; width: 30%;">
        <img src="{aws_path}/PRESENTACION+PUCE/yo.png" alt="Eduardo Cepeda" style="width: 100%; height: auto; border-radius: 8px;">
    </div>    
</section>
--
## Trayectoria

<img src="{aws_path}/PRESENTACION+PUCE/Presentation1.png" alt="trayectoria" style="width: 1500px; height: auto;">

---
<!-- .slide: data-background-color="#1a252f" -->

<div style="color: white; font-size: 0.75em;">

## <span style="color: #E67E22;">Plan de la Charla</span>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 20px 0;">

<div>
    <h3 style="color: #E67E22; margin-bottom: 15px;">Fundamentos</h3>
    <div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498DB; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Desafío actual:</strong> Investigación jurídica tradicional
        </p>
    </div>
    <div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ECC71; padding: 15px; border-radius: 5px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Revolución RAG:</strong> Búsqueda semántica de jurisprudencia
        </p>
    </div>
</div>

<div>
    <h3 style="color: #E67E22; margin-bottom: 15px;">Aplicaciones</h3>
    <div style="background: rgba(155, 89, 182, 0.1); border-left: 4px solid #9B59B6; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Ejemplos prácticos:</strong> Derecho ecuatoriano
        </p>
    </div>
    <div style="background: rgba(231, 76, 60, 0.1); border-left: 4px solid #E74C3C; padding: 15px; border-radius: 5px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Implementación:</strong> Consejo de la Judicatura
        </p>
    </div>
</div>

</div>

<div style="background: rgba(230, 126, 34, 0.1); border-left: 4px solid #E67E22; padding: 20px; margin: 20px 0; border-radius: 5px;">
    <h3 style="color: #E67E22; margin-bottom: 15px;">Objetivo Principal</h3>
    <p style="margin: 0; color: #BDC3C7;">
        <strong>De horas a minutos:</strong> Reducir la investigación jurídica y democratizar el acceso a la jurisprudencia ecuatoriana mediante Inteligencia Artificial.
    </p>
</div>

</div>

---
<!-- .slide: data-background-color="#1a252f" -->

<div style="color: white; font-size: 0.75em;">

## <span style="color: #E67E22;">El Desafío Actual: Investigación Jurídica en Ecuador</span>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 20px 0;">

<div>
    <h3 style="color: #E67E22; margin-bottom: 15px;">Problemas Potenciales</h3>
    <div style="background: rgba(231, 76, 60, 0.1); border-left: 4px solid #E74C3C; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Volumen masivo:</strong> Miles de sentencias anuales
        </p>
    </div>
    <div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498DB; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Tiempo excesivo:</strong> 4-8 horas por investigación
        </p>
    </div>
    <div style="background: rgba(155, 89, 182, 0.1); border-left: 4px solid #9B59B6; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Inconsistencias:</strong> Criterios diferentes entre juzgados
        </p>
    </div>
    <div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ECC71; padding: 15px; border-radius: 5px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Barreras de acceso:</strong> Ciudadanía sin acceso directo
        </p>
    </div>
</div>

<div>
    <h3 style="color: #E67E22; margin-bottom: 15px;">Impacto en el Sistema</h3>
    <div style="background: rgba(230, 126, 34, 0.1); border-left: 4px solid #E67E22; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Retrasos:</strong> Administración de justicia lenta
        </p>
    </div>
    <div style="background: rgba(231, 76, 60, 0.1); border-left: 4px solid #E74C3C; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Desigualdad:</strong> Acceso limitado a información jurídica
        </p>
    </div>
    <div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498DB; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Falta de uniformidad:</strong> Criterios judiciales inconsistentes
        </p>
    </div>
    <div style="background: rgba(155, 89, 182, 0.1); border-left: 4px solid #9B59B6; padding: 15px; border-radius: 5px;">
        <p style="margin: 0; color: #BDC3C7; font-size: 0.9em;">
            <strong>Sobrecarga:</strong> Trabajo excesivo en juzgados
        </p>
    </div>
</div>

</div>

<div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ECC71; padding: 20px; margin: 20px 0; border-radius: 5px;">
    <h3 style="color: #2ECC71; margin-bottom: 15px;">Recursos del Consejo</h3>
    <p style="margin: 0; color: #BDC3C7;">
        <strong>Portal de Estadística:</strong> <a href="https://fsweb.funcionjudicial.gob.ec/estadisticas/datoscj/portalestadistica.html" target="_blank" style="color: #3498DB;">Ver estadísticas</a> | 
        <strong>Servicios en Línea:</strong> <a href="https://www.funcionjudicial.gob.ec/servicios-en-linea/" target="_blank" style="color: #3498DB;">SATJE, SUPA, Formularios</a>
    </p>
</div>

</div>

---
<!-- Qué es RoPE self-extended -->
<!-- 4-8K tokens ~ 12 pages -> 1M ~ 1k pages -->
### Motivación: Mejoras en capacidad 

<img src="{aws_path}/PRESENTACION+PUCE/09-context-window.png" alt="Context Windows" style="width: 900px; height: auto;">

- La jurisprudencia ecuatoriana es un tesoro de conocimiento jurídico <br> <!-- .element: class="fragment" data-fragment-index="0" -->
- ... pero está puede estar dispersa y es difícil de consultar eficientemente <!-- .element: class="fragment" data-fragment-index="1" -->
- Con RAG, podemos "inyectar" toda la jurisprudencia en un sistema inteligente <!-- .element: class="fragment" data-fragment-index="2" -->
--
### Nuevo paradigma: 
LLM como Sistema Operativo

<img src="{aws_path}/PRESENTACION+PUCE/01-RAG-as-a-OS.png" alt="LLMs as OS" style="width: 900px; height: auto;"><!-- .element: class="fragment" data-fragment-index="0" -->
- RAG puede reemplazar/completar el fine-tunning <!-- .element: class="fragment" data-fragment-index="1" -->
---
**R**etrieval **A**ugmented **G**eneration

<img src="{aws_path}/PRESENTACION+PUCE/02-Schema.png" alt="schema" style="width: 900px; height: auto;">
---
## Perspectiva

| Nivel Básico                        | Avanzado                               | 
|-------------------------------------|----------------------------------------|
|                                     | - Transformación de preguntas          |
|                                     | - Ruteo                                |
|                                     | - Construcción de preguntas            |
| - Indexación                        | - Indexación                           |
| - Recuperación                      | - Recuperación                         |
| - Generación                        | - Generación                           |

---
<img src="{aws_path}/PRESENTACION+PUCE/03-Document-Loading.png" alt="LLMs as OS" style="width: 900px; height: auto;">
--
<img src="{aws_path}/PRESENTACION+PUCE/04-Numerical-representation.png" alt="numerical representation" style="width: 900px; height: auto;">
--
<img src="{aws_path}/PRESENTACION+PUCE/05-Loading-splitting-embedding.png" alt="load-splitting" style="width: 900px; height: auto;">
--
<img src="{aws_path}/PRESENTACION+PUCE/07-Vectorestore.png" alt="vectorstore" style="width: 900px; height: auto;">
---
### Manos a la obra:
- Modelo `distilbert-base-multilingual-cased` 
  - Fuente (checkpoints): <a target="_blank" href="https://huggingface.co/distilbert/distilbert-base-multilingual-cased">HuggingFace 🤗</a>
  - Información:  <a target="_blank" href="https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation">GitHub <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" style="width: 50px; height: auto;"></a>

- DistilBERT es un modelo multilingüe que soporta 104 idiomas.
- Tiene 6 capas, de dimensión 768 y 12 cabezas de atención.
- 134 millones de parámetros (vs 177 millones de mBERT-base).
- DistilBERT es el doble de rápido que mBERT-base.
--
### Aparte técnico
<img src="{aws_path}/PRESENTACION+PUCE/Prod-optim.png" alt="Production Optimisation" style="width: 700px; height: auto;">
<p style="font-size: 0.8em; text-align: center;">
  Para espíritus curiosos: <a target="_blank" href="https://medium.com/@quocnle/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26">Medium</a>
</p>
<section>
  <h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Optimización en producción</h3>
  <ul style="font-size: 0.9em;">
    <li><strong>Distilación:</strong> Entrenamiento supervisado de un modelo más pequeño.</li>
    <li><strong>Quantización:</strong> Reducir la precisión de los pesos - reducción de memoria.</li>
    <li><strong>Pruning:</strong> Eliminar conexiones o pesos irrelevantes.</li>
  </ul>
</section>
---
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Estructura de un modelo LLMs</h3>
<img src="{aws_path}/PRESENTACION+PUCE/08-trainable-parameters.png" alt="LLM structure" style="width: 800px; height: auto;">

  <h3 style="font-size: 0.9em; margin-left: 50px; text-align: left; color: #007BFF; text-transform: none;">Recursos Importantes</h3>

  <p style="font-size: 0.7em; margin-left: 50px; text-align: left;">Artículo original: 
    <a target="_blank" href="https://arxiv.org/abs/1706.03762" style="font-size: 0.7em; text-align: left; color: #FF5733; text-decoration: none;">
      Vaswani, A. "Attention is all you need." Advances in Neural Information Processing Systems (2017)
    </a>
  </p>

  <div style="font-size: 0.9em; text-align: left; list-style-type: disc; margin-left: 50px;">
      <ul style="font-size: 0.9em; text-align: left; list-style-type: disc; margin-left: 5px;">
        <li>
          <a target="_blank" href="https://nlp.seas.harvard.edu/annotated-transformer/" style="font-size: 0.9em; text-align: left; color: #3498DB; text-decoration: none;">
            Transformers Anotados
          </a>
        </li>
        <li>
          <a target="_blank" href="https://www.oreilly.com/library/view/natural-language-processing/9781098136789/" style="font-size: 0.9em; text-align: left; color: #3498DB; text-decoration: none;">
            Natural Language Processing 🤗
          </a>
        </li>
        <li>
          <a target="_blank" href="https://www.packtpub.com/en-us/product/transformers-for-natural-language-processing-9781803247335" style="font-size: 0.9em; text-align: left; color: #3498DB; text-decoration: none;">
            Transformers for NLP
          </a>
        </li>
      </ul>
  </div>
--

<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Completar la frase [MASK]</h3>
<pre><code class="language-python" data-line-numbers="1-2|3-5|6-9|10-12">import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
model_str = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_str)
model = AutoModelForMaskedLM.from_pretrained(model_str)
text = "El derecho de [MASK] está consagrado en la Constitución"
encoded_input = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    model_output = model(**encoded_input)
masked_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
logits = model_output.logits[0, masked_index, :]
top_5_token_ids = logits.topk(5, dim=-1).indices[0].tolist()
</code></pre>

--
<div>
  <h3 style="font-size: 1.1em; color: #007BFF; text-transform: none;">Texto Original:</h3>
  <p style="font-size: 1.0em; font-weight: bold; color: #FF5733;">El derecho de [MASK] está consagrado en la Constitución</p>
  <h3 style="font-size: 1.1em; color: #007BFF; text-transform: none;">Top 5 Predicciones</h3>
  <ul style="font-size: 1.0em;">
    <li>El derecho de <span style="color: #3498DB;">voto</span> está consagrado en la Constitución</li>
    <li>El derecho de <span style="color: #E74C3C;">Cristo</span> está consagrado en la Constitución</li>
    <li>El derecho de <span style="color: #2ECC71;">derecho</span> está consagrado en la Constitución</li>
    <li>El derecho de <span style="color: #F39C12;">libertad</span> está consagrado en la Constitución</li>
    <li>El derecho de <span style="color: #9B59B6;">Dios</span> está consagrado en la Constitución</li>
  </ul>
</div>
---
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Generadores de Features</h3>

<img src="{aws_path}/PRESENTACION+PUCE/06-transformers-as-feature-extractors.png" alt="LLM Features" style="width: 800px; height: auto;">

<p style="font-size: 0.9em; text-align: left; margin-left: 50px;">Artículo original: 
<a target="_blank" href="https://arxiv.org/abs/1706.03762" style="font-size: 0.7em; text-align: left; color: #FF5733; text-decoration: none;">
  Vaswani, A. "Attention is all you need." Advances in Neural Information Processing Systems (2017)
</a>
</p>
--
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Calcular embeddings</h3>
<pre><code class="language-python" data-line-numbers="1-3|4-7|8">import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
encoded_input = tokenizer(text_, return_tensors="pt")
with torch.no_grad():
    model_output = model(**encoded_input)
embedding_1 = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
similarity = cosine_similarity([text_reference_embeddings], [embedding_1])[0][0]
</code></pre>
--
<div>
    <h3 style="font-size: 0.8em; color: #007BFF;">Derecho Penal</h3>
    <ul style="font-size: 0.6em;">
      <li>El funcionario público que comete corrupción mediante cohecho debe enfrentar responsabilidad penal por el delito cometido.</li>
      <li>La corrupción de funcionarios públicos constituye delito que vulnera la confianza ciudadana y requiere sanción penal ejemplar.</li>
      <li>El cohecho de funcionarios públicos genera responsabilidad penal cuando se obtiene beneficio económico mediante corrupción.</li>
    </ul>
</div>
<div>
    <h3 style="font-size: 0.8em; color: #FF5733;">Derecho Civil</h3>
    <ul style="font-size: 0.6em;">
      <li>El funcionario público que comete corrupción debe responder civilmente por daños y perjuicios causados a terceros.</li>
      <li>La corrupción de funcionarios públicos genera responsabilidad civil extracontractual por daños y perjuicios.</li>
      <li>El funcionario corrupto debe indemnizar civilmente por los daños y perjuicios causados por su conducta.</li>
    </ul>
</div>
    <h3 style="font-size: 0.8em; color: #8E44AD;">Derecho Constitucional</h3>
    <ul style="font-size: 0.6em;">
      <li>El funcionario público que comete corrupción vulnera el derecho constitucional al debido proceso y la confianza ciudadana.</li>
      <li>La corrupción de funcionarios públicos constituye vulneración de derechos constitucionales fundamentales del Estado.</li>
      <li>El funcionario corrupto vulnera derechos constitucionales al debido proceso y la integridad de la administración pública.</li>
    </ul>
---
## Resultados

<p style="font-size: 0.7em; text-align: left;">"El funcionario público que comete corrupción vulnerando el derecho constitucional al debido proceso debe responder civilmente por daños y perjuicios, además de enfrentar responsabilidad penal por el delito de cohecho."</p>

$\scriptsize \rm{{cosine\\,similarity}} = \frac{{ Emb_1 \cdot Emb_2 }} {{ ||Emb_1|| \\, ||Emb_2|| }}$

<img src="{aws_path}/PRESENTACION+PUCE/similaridad_sematica.png?v=2024" alt="LLM Features" style="width: 600px; height: auto;">



--
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Generadores de Features</h3>

<img src="{aws_path}/PRESENTACION+PUCE/embeddings_projection.png?v=2024" alt="projections" style="width: 900px; height: auto;">
---
**R**etrieval **A**ugmented **G**eneration

<img src="{aws_path}/PRESENTACION+PUCE/02-Schema.png" alt="schema" style="width: 900px; height: auto;">
--
<!-- .slide: data-background-image="{aws_path}/PRESENTACION+PUCE/RAG-FULL.png" -->
---

## Aplicaciones Prácticas: RAG en el Sistema Judicial

### El Problema: Investigación Jurídica Tradicional

<div style="display: flex; justify-content: space-between; align-items: center;">
    <div style="width: 60%;">
        <h3 style="color: #FF5733;">Desafíos Actuales</h3>
        <ul style="font-size: 0.9em;">
            <li><strong>Búsqueda manual:</strong> Consulta en repositorios dispersos</li>
            <li><strong>Tiempo excesivo:</strong> 4-8 horas por investigación</li>
            <li><strong>Inconsistencias:</strong> Criterios diferentes entre juzgados</li>
            <li><strong>Barreras de acceso:</strong> Ciudadanía sin acceso directo</li>
        </ul>
    </div>
    <div style="width: 35%; text-align: center;">
        <div style="background-color: #f8f9fa; padding: 1em; border-radius: 8px;">
            <h4 style="color: #007BFF; margin: 0;">Impacto</h4>
            <p style="font-size: 0.8em; margin: 0.5em 0;">Retrasos en justicia</p>
            <p style="font-size: 0.8em; margin: 0.5em 0;">Desigualdad de acceso</p>
            <p style="font-size: 0.8em; margin: 0.5em 0;">Sobrecarga judicial</p>
        </div>
    </div>
</div>

---
<!-- .slide: data-background-color="#f8f9fa" -->

<div style="color: #2c3e50; font-size: 0.75em;">

## <span style="color: #098DFA;">La Solución: RAG Aplicado a Jurisprudencia</span>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 20px 0;">

<div>
    <h3 style="color: #098DFA; margin-bottom: 15px;">Antes</h3>
    <div style="background: rgba(231, 76, 60, 0.1); border-left: 4px solid #E74C3C; padding: 20px; border-radius: 5px; text-align: center;">
        <p style="margin: 0; color: #2c3e50; font-size: 1.2em; font-weight: bold;">
            Búsqueda Manual
        </p>
        <p style="margin: 10px 0; color: #E74C3C; font-size: 2em; font-weight: bold;">
            4-8 horas
        </p>
        <p style="margin: 0; color: #7f8c8d; font-size: 0.9em;">
            Resultados limitados
        </p>
    </div>
</div>

<div>
    <h3 style="color: #098DFA; margin-bottom: 15px;">Después</h3>
    <div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ECC71; padding: 20px; border-radius: 5px; text-align: center;">
        <p style="margin: 0; color: #2c3e50; font-size: 1.2em; font-weight: bold;">
            Búsqueda Semántica
        </p>
        <p style="margin: 10px 0; color: #2ECC71; font-size: 2em; font-weight: bold;">
            < 5 minutos
        </p>
        <p style="margin: 0; color: #7f8c8d; font-size: 0.9em;">
            Resultados precisos
        </p>
    </div>
</div>

</div>

<div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498DB; padding: 20px; margin: 20px 0; border-radius: 5px; text-align: center;">
    <h3 style="color: #3498DB; margin-bottom: 15px;">🚀 Transformación Digital</h3>
    <p style="margin: 0; color: #2c3e50; font-size: 1.1em;">
        <strong>De horas a minutos:</strong> La IA revoluciona la investigación jurídica ecuatoriana
    </p>
</div>

</div>

---
<!-- .slide: data-background-color="#f8f9fa" -->

<div style="color: #2c3e50; font-size: 0.75em;">

## <span style="color: #098DFA;">Caso de Uso 1: Búsqueda de Jurisprudencia Similar</span>

<div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498DB; padding: 20px; margin: 20px 0; border-radius: 5px;">
    <h3 style="color: #3498DB; margin-bottom: 15px;">Consulta del Usuario</h3>
    <p style="margin: 0; color: #2c3e50; font-style: italic; font-size: 1.1em;">
        "casos de violación de derecho al debido proceso en juicios laborales"
    </p>
</div>

<div style="display: grid; grid-template-columns: 2fr 1fr; gap: 30px; margin: 20px 0;">

<div>
    <h3 style="color: #098DFA; margin-bottom: 15px;">Resultados (Top 3)</h3>
    <div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ECC71; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #2c3e50; font-size: 0.9em;">
            <strong>Sentencia 123-2024-LA:</strong> "El derecho al debido proceso laboral incluye garantías de audiencia y defensa..."
        </p>
    </div>
    <div style="background: rgba(155, 89, 182, 0.1); border-left: 4px solid #9B59B6; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; color: #2c3e50; font-size: 0.9em;">
            <strong>Sentencia 089-2024-LA:</strong> "La violación del debido proceso en despidos genera nulidad del acto..."
        </p>
    </div>
    <div style="background: rgba(230, 126, 34, 0.1); border-left: 4px solid #E67E22; padding: 15px; border-radius: 5px;">
        <p style="margin: 0; color: #2c3e50; font-size: 0.9em;">
            <strong>Sentencia 156-2024-LA:</strong> "El empleador debe notificar previamente las causas del despido..."
        </p>
    </div>
</div>

<div>
    <div style="background: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ECC71; padding: 20px; border-radius: 5px; text-align: center; margin-bottom: 15px;">
        <h4 style="color: #2ECC71; margin: 0 0 10px 0;">Tiempo de Respuesta</h4>
        <p style="margin: 0; color: #2ECC71; font-size: 2em; font-weight: bold;">< 5 segundos</p>
    </div>
    <div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498DB; padding: 20px; border-radius: 5px; text-align: center;">
        <h4 style="color: #3498DB; margin: 0 0 10px 0;">Precisión</h4>
        <p style="margin: 0; color: #3498DB; font-size: 1.5em; font-weight: bold;">95% relevancia</p>
    </div>
</div>

</div>

</div>

---
### Caso de Uso 2: Búsqueda de Precedentes Aplicables

#### Consulta Especializada
<blockquote style="background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%); padding: 1.5em; border-left: 4px solid #098DFA; border-radius: 8px; box-shadow: 0 2px 8px rgba(9, 141, 250, 0.1);">
"precedentes sobre indemnización por daño moral en accidentes de tránsito"
</blockquote>

#### Doctrina Jurisprudencial Identificada
<div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.5em; border-radius: 12px; margin: 1em 0; border-left: 4px solid #FF9800; box-shadow: 0 2px 8px rgba(255, 152, 0, 0.1);">
<strong style="color: #E65100;">Ratio Decidendi:</strong> "La indemnización por daño moral se calcula considerando el impacto psicológico, la gravedad del accidente y la capacidad económica del responsable."
</div>

#### Sentencias de Referencia
- **Sentencia 234-2024-CV:** Criterio de cálculo por impacto psicológico
- **Sentencia 189-2024-CV:** Consideración de capacidad económica  
- **Sentencia 301-2024-CV:** Estándares de gravedad del accidente

<div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1.5em; border-radius: 12px; text-align: center; margin-top: 1em; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);">
<strong style="color: #2e7d32; font-size: 1.1em;">Beneficio:</strong> Identificación automática de doctrina aplicable
</div>

---
### Caso de Uso 3: Análisis de Consistencia Judicial

#### Detección de Contradicciones

<table style="width: 100%; margin: 1em 0; border-collapse: separate; border-spacing: 10px;">
<tr>
<td style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); padding: 1.5em; text-align: center; width: 50%; border-radius: 12px; border-left: 4px solid #f44336; box-shadow: 0 2px 8px rgba(244, 67, 54, 0.1);">
<h4 style="color: #d32f2f; margin: 0 0 10px 0;">Juzgado A</h4>
<p style="margin: 10px 0; font-style: italic;">"La pensión alimenticia se calcula como 30% del ingreso del obligado"</p>
<small style="color: #666;">Sentencia 123-2024-FA</small>
</td>
<td style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); padding: 1.5em; text-align: center; width: 50%; border-radius: 12px; border-left: 4px solid #f44336; box-shadow: 0 2px 8px rgba(244, 67, 54, 0.1);">
<h4 style="color: #d32f2f; margin: 0 0 10px 0;">Juzgado B</h4>
<p style="margin: 10px 0; font-style: italic;">"La pensión alimenticia se calcula como 25% del ingreso del obligado"</p>
<small style="color: #666;">Sentencia 456-2024-FA</small>
</td>
</tr>
</table>

<div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1.5em; border-radius: 12px; text-align: center; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);">
<h4 style="color: #2e7d32; margin: 0 0 10px 0; font-size: 1.1em;">⚠️ Sistema Detecta Inconsistencia</h4>
<p style="margin: 0.5em 0; font-weight: 500;">Alerta automática para unificar criterios entre juzgados</p>
</div>

---
### Caso de Uso 4: Democratización del Acceso

#### Consulta Ciudadana
<blockquote style="background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%); padding: 1.5em; border-left: 4px solid #098DFA; border-radius: 12px; box-shadow: 0 2px 8px rgba(9, 141, 250, 0.1); font-size: 1.1em;">
"¿Qué dice la ley sobre pensión alimenticia?"
</blockquote>

#### Respuesta del Sistema
<div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1.5em; border-radius: 12px; margin: 1em 0; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);">
<p style="margin: 0; font-style: italic; color: #2e7d32;">"La pensión alimenticia es un derecho fundamental que garantiza el sustento de hijos menores. Se calcula según la capacidad económica del obligado y las necesidades del beneficiario, conforme al Código Civil ecuatoriano."</p>
</div>

#### Información Adicional
- Procedimiento para solicitar pensión
- Documentos requeridos  
- Plazos legales aplicables
- Recursos disponibles

<table style="width: 100%; margin-top: 1em;">
<tr>
<td style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1.5em; text-align: center; border-radius: 12px; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);">
<h4 style="color: #2e7d32; margin: 0 0 10px 0; font-size: 1.2em;">Acceso Universal</h4>
<p style="margin: 0.5em 0; font-weight: 500; font-size: 1.1em;">24/7 disponible • Lenguaje comprensible • Sin costo</p>
</td>
</tr>
</table>

---
### Integración con Servicios del Consejo

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.2em; margin: 1.5em 0;">
    <div style="background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%); padding: 1em; border-radius: 8px; border-left: 3px solid #098DFA; box-shadow: 0 2px 6px rgba(9, 141, 250, 0.1);">
        <h4 style="color: #098DFA; margin: 0 0 6px 0; font-size: 0.95em;">SATJE</h4>
        <p style="font-size: 0.75em; margin: 0.3em 0; color: #2c3e50;">Sistema Automático de Trámites Judiciales</p>
        <ul style="font-size: 0.7em; color: #2c3e50; margin: 0.3em 0; padding-left: 1.2em;">
            <li>Búsqueda inteligente de precedentes</li>
            <li>Asistencia en redacción de sentencias</li>
        </ul>
    </div>
    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1em; border-radius: 8px; border-left: 3px solid #4CAF50; box-shadow: 0 2px 6px rgba(76, 175, 80, 0.1);">
        <h4 style="color: #2e7d32; margin: 0 0 6px 0; font-size: 0.95em;">SUPA</h4>
        <p style="font-size: 0.75em; margin: 0.3em 0; color: #2c3e50;">Sistema de Pensiones Alimenticias</p>
        <ul style="font-size: 0.7em; color: #2c3e50; margin: 0.3em 0; padding-left: 1.2em;">
            <li>Cálculo automático de pensiones</li>
            <li>Consulta de jurisprudencia aplicable</li>
        </ul>
    </div>
    <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1em; border-radius: 8px; border-left: 3px solid #FF9800; box-shadow: 0 2px 6px rgba(255, 152, 0, 0.1);">
        <h4 style="color: #E65100; margin: 0 0 6px 0; font-size: 0.95em;">Portal Estadístico</h4>
        <p style="font-size: 0.75em; margin: 0.3em 0; color: #2c3e50;">Análisis de tendencias judiciales</p>
        <ul style="font-size: 0.7em; color: #2c3e50; margin: 0.3em 0; padding-left: 1.2em;">
            <li>Patrones en sentencias</li>
            <li>Análisis de consistencia</li>
        </ul>
    </div>
    <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1em; border-radius: 8px; border-left: 3px solid #9C27B0; box-shadow: 0 2px 6px rgba(156, 39, 176, 0.1);">
        <h4 style="color: #7b1fa2; margin: 0 0 6px 0; font-size: 0.95em;">Servicios Ciudadanos</h4>
        <p style="font-size: 0.75em; margin: 0.3em 0; color: #2c3e50;">Acceso democrático a la justicia</p>
        <ul style="font-size: 0.7em; color: #2c3e50; margin: 0.3em 0; padding-left: 1.2em;">
            <li>Consultas en lenguaje natural</li>
            <li>Información jurídica accesible</li>
        </ul>
    </div>
</div>

---
<!-- .slide: data-visibility="hidden" -->
### Arquitectura del Sistema Judicial RAG

<div style="text-align: center;">
    <h3 style="color: #098DFA; margin-bottom: 2em;">Flujo de Datos y Procesamiento</h3>
    
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 2em 0; gap: 1em;">
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5em; border-radius: 12px; width: 20%; border-left: 4px solid #2196F3; box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);">
            <h4 style="color: #1976d2; margin: 0 0 10px 0; font-size: 1.1em;">Fuente</h4>
            <p style="font-size: 0.9em; margin: 0.5em 0; color: #2c3e50;">Base de datos de sentencias del Consejo</p>
        </div>
        <div style="color: #666; font-size: 1.5em; font-weight: bold;">→</div>
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1.5em; border-radius: 12px; width: 20%; border-left: 4px solid #9C27B0; box-shadow: 0 2px 8px rgba(156, 39, 176, 0.1);">
            <h4 style="color: #7b1fa2; margin: 0 0 10px 0; font-size: 1.1em;">Procesamiento</h4>
            <p style="font-size: 0.9em; margin: 0.5em 0; color: #2c3e50;">Extracción de texto de PDFs</p>
        </div>
        <div style="color: #666; font-size: 1.5em; font-weight: bold;">→</div>
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1.5em; border-radius: 12px; width: 20%; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);">
            <h4 style="color: #2e7d32; margin: 0 0 10px 0; font-size: 1.1em;">Indexación</h4>
            <p style="font-size: 0.9em; margin: 0.5em 0; color: #2c3e50;">Embeddings de sentencias</p>
        </div>
        <div style="color: #666; font-size: 1.5em; font-weight: bold;">→</div>
        <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.5em; border-radius: 12px; width: 20%; border-left: 4px solid #FF9800; box-shadow: 0 2px 8px rgba(255, 152, 0, 0.1);">
            <h4 style="color: #E65100; margin: 0 0 10px 0; font-size: 1.1em;">Consulta</h4>
            <p style="font-size: 0.9em; margin: 0.5em 0; color: #2c3e50;">Interface para usuarios</p>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1.5em; border-radius: 12px; margin: 2em 0; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);">
        <h4 style="color: #2e7d32; margin: 0 0 10px 0; font-size: 1.2em;">Respuesta Inteligente</h4>
        <p style="font-size: 1em; margin: 0.5em 0; color: #2c3e50; font-weight: 500;">Sentencias relevantes + explicación contextual</p>
    </div>
</div>

---
## El Futuro de la Justicia en Ecuador

<div style="text-align: center; margin: 2em 0;">
    <h3 style="color: #098DFA; margin-bottom: 2em;">Mensaje Central</h3>
    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 2.5em; border-radius: 16px; margin: 2em 0; border-left: 6px solid #4CAF50; box-shadow: 0 4px 16px rgba(76, 175, 80, 0.15);">
        <p style="font-size: 1.3em; font-style: italic; color: #2e7d32; margin: 0; line-height: 1.6;">
            "La IA no reemplaza al juez, sino que potencia su capacidad de encontrar jurisprudencia relevante, 
            permitiendo dedicar más tiempo al análisis y la justicia, y menos a la búsqueda manual."
        </p>
    </div>
    
    <h3 style="color: #098DFA; margin: 2em 0 1em 0;">Ecuador como Pionero</h3>
    <div style="background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%); padding: 2em; border-radius: 16px; border-left: 6px solid #098DFA; box-shadow: 0 4px 16px rgba(9, 141, 250, 0.15);">
        <p style="font-size: 1.2em; margin: 0; color: #2c3e50; line-height: 1.6;">
            Podemos ser líderes en IA judicial en la región, democratizando el acceso a la justicia 
            y mejorando la eficiencia del sistema judicial ecuatoriano.
        </p>
    </div>
</div>

---
<h3>Retroalimentación</h3>
<img src="{aws_path}/PRESENTACION+PUCE/flujo_final.png" alt="flujo_final" style="width: 9000px; height: auto;">

<img src="{aws_path}/PRESENTACION+PUCE/Fillout QR Code.png" alt="projections" style="width: 150px; height: auto;">
<a target="_blank" href="https://forms.fillout.com/t/tmNM7SUWuJus" style="font-size: 0.7em; text-align: left; color: #0772CA; text-decoration: none;">
  https://forms.fillout.com/t/tmNM7SUWuJus
</a>

---

## The end
<img src="{aws_path}/PRESENTACION+PUCE/cari.png" alt="projections" style="width: 700px; height: auto;">
<div style="font-size: 0.9em; text-align: left; list-style-type: disc; margin-left: 200px;">
<ul>
<li>🌍 <a target="_blank" href="https://cepeda.fr">cepeda.fr</a></li> 
<li><img src="{aws_path}/PRESENTACION+PUCE/LILOGO.png" alt="LinkedIn Logo" style="width: 40px; height: auto; margin-right: 15px;"> <a target="_blank" href="https://www.linkedin.com/in/educep/">/educep</a></li>
<li>✉️ <a href="mailto:eduardo@cepeda.fr">eduardo@cepeda.fr</a></li>
<li><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" style="width: 40px; height: auto; margin-right: 15px;"> <a target="_blank" href="https://github.com/educep/judicatura">GitHub /funcion_judicial</a></li>
<li><img src="{aws_path}/DS_Imagen_de_marca/logos/DS+logo.png" alt="DatosCout Logo" style="width: 40px; height: auto; margin-right: 15px;"> <a target="_blank" href="https://funcion_judicial.datoscout.ec">funcion_judicial.datoscout.ec</a></li>
</ul>
</div>
"""