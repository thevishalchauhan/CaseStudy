# CaseStudy
## Thought Process
- From the raw notebook I took the best model LightGBM
- Saved the model weight under ./weights
- Created the strealit app for UI for inference
- Dockerized the git using DockerFile



## Aprroaches
### Approach 1
- Clone the repo
- Build Image
  - docker build -t ImageName .
- Run the Container [__8080__ port for streamlit]
  - docker run --rm -d -p 8000:8080 --name ContainerName ImageName
- Streamlit UI for inference
![Image](https://github.com/vishal0143/AdidasCaseStudy/blob/main/imgs/ui.png "StreamLit UI")

### Approach 2
- Pull my image from docker hub 
  - docker pull vischauh/casestudy
- Run the container
  - docker run -d --name case --rm -p 8000:8080 vischauh/casestudy
