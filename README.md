# AdidasCaseStudy

## Thought Process
- Observing the raw notebook and taking best model -> LightGBM
- Stored the weights under /weight
- Created a streamlit application for inference
![Image](https://github.com/vishal0143/AdidasCaseStudy/blob/main/imgs/ui.png "StreamLit UI")
- Dockerized the git using Docker File
- Build -> docker build -t <imagename> .
- Run -> docker run --rm -d -p 8000:8080 <imagename>

### OR
- Pull my image from docker hub -> "docker pull vischauh/casestudy"
- Run the container
