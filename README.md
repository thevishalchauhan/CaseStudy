# AdidasCaseStudy

## Thought Process
- Observing the raw notebook and taking best model -> LightGBM
- Stored the weights under /weight
- Created a streamlit application for inference
![Image](https://github.com/vishal0143/AdidasCaseStudy/blob/main/imgs/ui.png "StreamLit UI")
- Dockerized the git using Docker File
- Build -> docker build -t ImageName .
- Run -> docker run --rm -d -p 8000:8080 --name ContainerName ImageName

### OR
- Pull my image from docker hub -> "docker pull vischauh/casestudy"
- Run the container
