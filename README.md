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
Follow instructions in the submit tab of the EvalAI challenge page to submit your docker image. Note that you will need a version of EvalAI >= 1.2.3. Pasting those instructions here for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.2.3"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push my_submission:latest --phase <phase-name>
```

Valid challenge phases are gibson20-{minival, testsim, testreal}-{static, interactive, dynamic}.

The challenge consists of the following phases:

1. Minival: The purpose of this phase is mainly for sanity checking.  The participants are given a Train and Validation sets to train and evaluate policies. These sets represent environments similar to the final real deployment environment, and model a LoCoBot, the platform which is used in the final evaluation. Participants can submit their policy for evaluation on the aforementioned Validation set to an evaluation server hosted by EvalAI.


2. Testsim: The participants are evaluated on a held-out Test environment, which mimics the final real world test but isn't identical to it. It is a scan of the real apartment where the final challenge is held. However, the furniture, cluttering and dynamic objects are arranged differently.  
Further, this test is used to rank all participants. Only top 5 participants are to be considered for the real world evaluation.

3. Testreal: In this step we are given the opportunity to the participants to develop on the real robot. In more detail, we will allow the participants to run up to X times their policy in the real apartment. The furniture arrangement and objects are different from the ones used in the final test. The participants are given videos of their runs and evaluation metrics, which are intended to facilitate debugging. In the final step the participating policies are tested on a real robot in a real environment. It is the same environment as in Step 3, however furniture and objects are arranged differently, and a new bedroom and a new bathroom are revealed. This test is used to rank participants and determine a winner.


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