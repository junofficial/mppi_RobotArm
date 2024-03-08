<!-- ABOUT THE PROJECT -->
## About The Project

MPPI 제어 기법을 통한 로봇암을 제어하는 알고리즘을 제작해보려 합니다
<br>

![Screenshot from 2024-03-06 00-33-52](https://github.com/junofficial/IsaacGym_Install/assets/124868359/6e0b5725-a817-4781-acad-b971808e17a1)

<br>
Isaac Gym의 경우 Nvidia에서 제공하는 로봇 공학을 위한 플랫폼으로 위에 첨부한 Nvidia의 Isaac Gym 사이트에서 가입과 권한 허가 이후 Isaac Gym 파일을 다운 받을 수 있습니다.
<br>

<p align="center">
  <img src="https://github.com/junofficial/IsaacGym_Install/assets/124868359/9000fed9-87ad-4bfb-95d7-ef00ccd72f27">
</p>


정상적으로 진행하셨다면 이러한 파일을 다운로드 받을 수 있습니다. 다운로드한 파일을 압축해제 해주시고 설치를 진행하게 됩니다.



<!-- GETTING STARTED -->
## Getting Started

이 repo에서는 오직 Isaacgym을 설치하는 부분만 작성하였습니다. Isaacgym을 통해서 legged_gym과 storm 환경을 사용해보았는데 관련 내용은 밑에 링크를 통해서 첨부해놓겠습니다.

### Installation
먼저 
```sh
conda create -n IsaacGym python=3.7
```
을 통해 파이썬 3.7의 환경을 create와 activate 해주시고 설치를 진행하겠습니다. 

1. Cuda toolkit 설치(본인 컴퓨터에 맞는 버전으로 선택):
![Screenshot from 2024-03-06 00-27-09](https://github.com/junofficial/IsaacGym_Install/assets/124868359/cab75001-280b-4ec3-a294-24520cb88cfd)
2. pytorch 1.12.1과 cuda-11.6 설치:
   ```sh
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```
3. 압축해제한 isaacgym 파일을 대상으로 밑의 명령어 실행: 
   ```sh
   cd isaacgym/python && pip install -e .
   ```
4. isaacgym이 설치됬는지 확인하기 위해 example 실행:
   ```sh
   cd examples && python 1080_balls_of_solitude.py
   ```
   Ubuntu 20.04 기준 example을 실행하게 되면 "ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory"라는 오류가 발생하게 되는 데(설치가이드 문서 install.html에 명시) 이 오류를 해결하는 법도 함께 첨부합니다. 이러한 오류는 LD_LIBRARY_PATH 환경변수를 새로 만든 아나콘다 가상환경에 대한 경로로 설정해주어야 하기 때문에 발생하게 됩니다.
   ```sh
   cd /home/<user>/anaconda3/envs/py37
   mkdir -p ./etc/conda/activate.d
   mkdir -p ./etc/conda/deactivate.d
   touch ./etc/conda/activate.d/env_vars.sh
   touch ./etc/conda/deactivate.d/env_vars.sh
   ```
   이후 ./etc/conda/activate.d/env_vars.sh 파일을 열고
   ```sh
   export LD_LIBRARY_PATH=/home/<user>/anaconda3/envs/py37/lib
   ```
   또한 ./etc/conda/deactivate.d/env_vars.sh 파일을 열어
   ```sh
   unset LD_LIBRARY_PATH
   ```
   을 작성해주면 오류가 해결됩니다.

## Usage

설치한 Isaac Gym을 통해서 4족보행로봇을 Parallel Deep RL을 통해 학습시키는 legged_gym과 mppi방법을 통해서 로봇팔을 제어하는 storm을 사용해보았습니다. 관련 링크들은 밑에 첨부해뒀으며 추가 설치 내용은 첨부된 github에 작성된 내용을 통해 진행하시면 됩니다.

https://github.com/leggedrobotics/legged_gym
<br>
https://github.com/NVlabs/storm?tab=readme-ov-file
