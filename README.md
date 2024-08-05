# Gemini_use_case

This project utilizes the Gemini-pro model created by Google and the RAG (Retrieval-Augmented Generation) technique. It is applied to the Systems Programming course at Universidad Carlos III de Madrid (UC3M). This course is fundamental in training students in the Java programming language.
The model’s responses are evaluated in two types of questions: multiple-choice questions and code generation problems. 
For both cases, Gemini-pro 1.5 was used to extract course content (pdf) to provide as context, along with text extraction to complement the content for RAG. Additionally, for multiple-choice exercises,
previous exams were also tested as context.

To use Vertex AI Generative AI you must have the langchain-google-vertexai Python package installed and either:

Have credentials configured for your environment (gcloud, workload identity, etc...)
Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable

For more information, see:

https://cloud.google.com/docs/authentication/application-default-credentials#GAC
https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth
