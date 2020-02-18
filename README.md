Sim2Real Challenge with Gibson @ CVPR 2020
===================================
This repository contains starter code for sim2real challenge with Gibson. For an overview of the challenge, visit [http://svl.stanford.edu/gibson2/challenge](http://svl.stanford.edu/gibson2/challenge) .

Task
----------------------------
PointNav, PointNav with Interactable objects, PointNav with dynamic obstacles. Details TBA.

Challenge Dataset
----------------------------

We used navigation episodes Habitat created as the main dataset for the challenge. In addition, we scanned a 
new house named "Castro" and use part of it for training. The evaluation will be in Castro in both sim and real.

- Training scenes: 106 Gibson Scenes + Castro
- Dev scenes: Castro(Sim), Castro(Real)
- Evaluation scenes: CastroFull(Sim), CastroFull(Real)


Evaluation
-----------------------------
After calling the STOP action, the agent is evaluated using the "Success weighted by Path Length" (SPL) metric [3].

![SPL](./misc/spl.png)

An episode is deemed successful if on calling the STOP action, the agent is within 0.36m of the goal position. The evaluation will be carried out in completely new houses which are not present in training and validation splits.

Participation Guidelines
-----------------------------
Participate in the contest by registering on the EvalAI challenge page and creating a team. Participants will upload docker containers with their agents that evaluated on a AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to make sure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation
- Step 1: Clone the challenge repository
```bash
git clone https://github.com/StanfordVL/GibsonSim2RealCallenge.git
cd GibsonSim2RealCallenge
```

Implement your own agent, one example is a random agent in `agent.py`.

```python3
from gibson2.envs.challenge import Challenge
import numpy as np

class RandomAgent:
    def reset(self):
        pass

    def act(self, observations):
        action = np.random.uniform(low=-1,high=1,size=(2,))
        return action

def main():
    agent = RandomAgent()
    challenge = Challenge()
    challenge.submit(agent)

if __name__ == "__main__":
    main()
```

- Step 2: Install nvidia-docker2, following the [guide](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0). 

- Step 3: Modify the provided Dockerfile to accommodate any dependencies. A minimal Dockerfile is shown below.
```Dockerfile
FROM gibsonchallenge/gibsonv2:latest
ENV PATH /miniconda/envs/gibson/bin:$PATH

ADD agent.py /agent.py
ADD submission.sh /submission.sh
WORKDIR /
```
- Step 4: 

Download challenge data from [here](https://docs.google.com/forms/d/e/1FAIpQLSen7LZXKVl_HuiePaFzG_0Boo6V3J5lJgzt3oPeSfPr4HTIEA/viewform) and put in `GibsonSim2RealCallenge/gibson-challenge-data`

- Step 5:

Evaluate locally:

You can run `./test_locally.sh --docker-name my_submission`

If things work properly, you should be able to see the terminal output in the end:
```
...
episode done, total reward -0.31142135623731104, total success 0
episode done, total reward -5.084213562373038, total success 0
episode done, total reward -11.291320343559496, total success 0
episode done, total reward -16.125634918610242, total success 0
episode done, total reward -16.557056274847586, total success 0
...
```

### Online submission
TBA

### Starter code and Training
TBA

Acknowledgments
-------------------
We thank habitat team for the effort of converging task setup and challenge API. 


References 
-------------------
[1] Interactive Gibson: A Benchmark for Interactive Navigation in Cluttered Environments.  Xia, Fei, William B. Shen, Chengshu Li, Priya Kasimbeg, Micael Tchapmi, Alexander Toshev, Roberto Martín-Martín, and Silvio Savarese. arXiv preprint arXiv:1910.14442 (2019).

[2] Gibson env: Real-world perception for embodied agents. F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese. In CVPR, 2018

[3] On evaluation of embodied navigation agents. Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, Amir R. Zamir. arXiv:1807.06757, 2018.