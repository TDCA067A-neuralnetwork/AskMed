# Clinical Question Answering System using Medical LLMs

TYBCA Neural Network project.

## Core architecture

User
 -> Input Validation
 -> TF-IDF
 -> MLP Neural Network Medical/Non-Medical Classifier
 -> Medical LLM
 -> Safety response
 -> User

The LLM is never called when the classifier rejects the query.

## Run locally

Windows:

    py -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

Then open:

    http://127.0.0.1:5000

Run tests:

    pytest -q

## Configure the Medical LLM

The web application is separated from the large model. This avoids trying to
load a multi-billion-parameter model inside a small Vercel Python function.

Set these environment variables:

    HF_BASE_URL=<your OpenAI-compatible inference endpoint>
    HF_TOKEN=<your secret token>
    MEDICAL_MODEL=<your selected medical model>

Do not commit tokens to GitHub.

If these variables are missing, the classifier still works, but accepted
medical questions display a configuration message instead of pretending that
an LLM answer was generated.

## Vercel

Vercel runs the Flask API and web interface. Put the actual model behind a
hosted inference endpoint and store credentials in Vercel Environment
Variables.

## Academic evaluation

The included dataset is a starter dataset, not a clinical benchmark.

For the final report:
- expand the dataset,
- create separate train/test splits,
- report accuracy,
- precision,
- recall,
- F1-score,
- confusion matrix,
- test prompt-injection cases,
- test ambiguous medical/non-medical queries.

## Safety

This is an educational prototype. It is not a clinically validated medical
device and must not be used as a substitute for a doctor or emergency service.
