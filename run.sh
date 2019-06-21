# Make a POST request to the /classify command, receiving
curl http://0.0.0.0:8000/generate \
   -X POST \
   -H "content-type: application/json" \
   -d "{ \"z\": \"0.5\" }"