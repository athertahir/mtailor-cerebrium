**Deploy Classification Neural Network on Serverless GPU platform of Cerebrium**

Please make sure that python 3.10 and pip is already installed. Also, docker must be running locally.


**Setup**


1. Create virtual environment and install pip packages:

```
python -m venv .venv
source .venv/bin/activate

pip3 install -r requirements.txt 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. Optional: Run pytorch model (Original implementation):

`python pytorch_model.py`


3. To create `ONNX` file from pytorch model, run the following script:

`python convert_to_onnx.py`


4.  Run FastAPI server by creating and running the docker image:

```
sudo docker build -t mtailor-model .

sudo docker run --name test  -p 8192:8192 mtailor-model
```

![](./images/1.png)

**Output**

![](./images/2.png)


### Cerebrium Deployment

Create account on cerebrium and login via CLI:

```
pip install cerebrium

cerebrium login
```


Deploy the model from the code directory using following command:

`cerebrium deploy`

![](./images/3.png)

![](./images/4.png)

![](./images/5.png)

### Test Cerebrium Deployment

![](./images/6.png)

**Important** Make sure to run the script where image also exists:

```
python test_server.py   --image_path n01440764_tench.jpeg   --endpoint https://api.cortex.cerebrium.ai/v4/p-42825189/mtailor-model   --api_key "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTQyODI1MTg5IiwiaWF0IjoxNzQ5NTQ4NDc2LCJleHAiOjIwNjUxMjQ0NzZ9.2QTzFmf7wqfFakDXUJiO35nrv86dQPDyAB0IFlrh30Ki5C-UudPsDhz8jO-G1eFmcrYESF01D_fhEaoB5Hz8IfDH1LbT-hoHPV_deGzWUhqmIKeXrLXBpegONqDL76e-sTTMJkBi8HgEH2GwwmMwIJlNGAOA-0AzESjQC3RYKcKOzD5eATjCYQ0A3F5EuPRZUpW4WGHsgMkTR3Y9N888kzwl_011LwSAHsdUyWXo6Xbe8KcYo8I5Jl-UPo-_T5iRwr_BFWKLszBuYRiVOj93IujfkkS8VxiJKxyxtyqA9raTsfXTsFeLdxI9hP4VxsLnsIk4Mlx3rKEhML7Dx3k_gw"
```
![](./images/7.png)

**Code Description**
- convert_to_onnx.py | codebase to convert the PyTorch Model to the ONNX model
- model.py with the following classes/functionalities, make their separate classes:
    - Onnx Model loading and prediction call
    - Pre-processing of the Image [Sample code provided in pytorch_model.py]
- test.py | codebase to test the code/model written. This should test everything one would expect for ML Model deployment.
- test_server.py | codebase to make a call to the model deployed on the Cerebrium (Note: This should test deployment not something on your local machine)

