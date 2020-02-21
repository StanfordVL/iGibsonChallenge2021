Sim2Real Challenge with Gibson @ CVPR 2020
=============================================

This repository contains starter code for sim2real challenge with Gibson brought to you by [Stanford VL](http://svl.stanford.edu) and [Robotics @ Google](https://research.google/teams/brain/robotics/). 
For an overview of the challenge, visit [the challenge website](http://svl.stanford.edu/gibson2/challenge.html).

Task
----------------------------
The first Gibson Sim2Real Challenge is composed of three simultaneous tracks that model important skills in visual navigation:

- PointNav Track: the goal for an agent in this track is to successfully to 
a desired location based on visual information (RGB+D). In this track, the agent is not allowed to collide with the environment. 
This track will evaluate the sim2real transference of the most basic capability in a navigating agent. 
We will evaluate performance in this track using SPL [3].

- InteractiveNav Track: in this track the agent is allowed (even encouraged) 
to collide with the environment in order to push obstacles away. 
But careful! Some of the obstacles are not movable. This track evaluates 
agents in Interactive Navigation tasks [1], navigation problems that 
considers interactions with the environment. We will use INS [1] to 
evaluate performance of agents in this track.

- DynamicObstacleNav Track: the agents in this track need to avoid collisions 
with a dynamic agent with different unknown navigating patterns. 
Reasoning about the motion of other agents is challenging, 
and we will measure how well existing sim2real solutions perform in this conditions. 
No collisions are allowed in this track. We will use again SPL to evaluate the agents.

All submissions to our challenge will be evaluated in these three tracks, 
but we will announce separate winners for each of them.

Challenge Dataset
----------------------------

We used navigation episodes Habitat created as the main dataset for the challenge. In addition, we scanned a 
new house named "Castro" and use part of it for training. The evaluation will be in Castro in both sim and real.

- Training scenes: 572 Gibson Scenes + Castro
- Dev scenes: Castro(Sim), Castro(Real)
- Evaluation scenes: CastroFull(Sim), CastroFull(Real)


Evaluation
-----------------------------
After calling the STOP action, the agent is evaluated using the "Success weighted by Path Length" (SPL) metric [3].

<p align="center">
  <img src='misc/spl.png' />
</p>

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

- Step 2: Install nvidia-docker2, following the guide: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0). 

- Step 3: Modify the provided Dockerfile to accommodate any dependencies. A minimal Dockerfile is shown below.
  ```Dockerfile
  FROM gibsonchallenge/gibsonv2:latest
  ENV PATH /miniconda/envs/gibson/bin:$PATH

  ADD agent.py /agent.py
  ADD submission.sh /submission.sh
  WORKDIR /
  ```

  Build your docker container: `docker build . -t my_submission` , where `my_submission` is the docker image name you want to use.

- Step 4: 

  Download challenge data from [here](https://docs.google.com/forms/d/e/1FAIpQLSen7LZXKVl_HuiePaFzG_0Boo6V3J5lJgzt3oPeSfPr4HTIEA/viewform) and put in `GibsonSim2RealCallenge/gibson-challenge-data`.
  
  Also, change the directory permission.
  ```
  chmod -R 777 gibson-challenge-data/
  ```

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

Valid challenge phases are `gibson20-{minival, testsim, testreal}-{static, interactive, dynamic}`, 
where `static, interactive, dynamic` corresponds to three tracks.

Our Sim2Real Challenge consists of three phases:

- Phase 0: Sanity Check (`minival`): 
The purpose of this phase is mainly for sanity checking and make sure the policy 
can be successfully submitted and evaluated.
Participant can submit any policy, even trivial policy to our evaluation server. 
Each team is allowed maximum of 30 submission per day for this phase. 
We will block and disqualify teams that spam our servers.

- Phase 1: Simulation phase (`testsim`): 
In this phase participants will develop their solutions in our simulator Gibson. 
They will get access to all Gibson 3D reconstructed environments (572) and a 
3D reconstruction of part of the real world environment where this phase will take place, named Castro. 
Another part of the apartment will be kept unseen to perform evaluation (CastroFull). 
Participants can submit their solutions at any time through the [evalai](https://evalai.cloudcv.org) portal. 
At the end of the classification time (TBA), the best 5 solutions will pass to the second phase, the real world.

- Phase 2: Real World phase (`testreal`): The classified teams will receive 30 min/day to 
evaluate their policies on our real world platform. The runs will be saved as footages for debugging. 
They will also receive a record of the states, measurements, 
and actions taken by the real world agent at the end of each run, 
as well as the score. The last two days (TBA) are classification time: 
the teams will be ranked by their scores. 
At the end of these two days we will announce the winner of the first Gibson Sim2Real Challenge.

- Phase 3: Demo phase: The best three entries 
of our challenge will receive the opportunity to 
showcase their solutions live during CVPR20! We will connect 
directly from Seattle and video stream a run of each solutions, 
highlighting their strengths and characteristics. 
This will provide an opportunity for the teams to explain their solution.


### Training

#### Using Docker
`./train_locally.sh --docker-name my_submission`

#### Not using Docker
- Step 0: install [anaconda](https://docs.anaconda.com/anaconda/install/) and create a python3.6 environment
  ```
  conda create -n gibson python=3.6
  conda activate gibson
  ```
- Step 1: install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html). We tested with CUDA 10.0 and 10.1 and cuDNN 7.6.5

- Step 2: install EGL dependency
  ```
  sudo apt-get install libegl1-mesa-dev
  ```
- Step 3: install [GibsonEnvV2](https://github.com/StanfordVL/GibsonEnvV2) and download Gibson [assets](https://storage.googleapis.com/gibsonassets/assets_gibson_v2.tar.gz) and [dataset](https://docs.google.com/forms/d/e/1FAIpQLSen7LZXKVl_HuiePaFzG_0Boo6V3J5lJgzt3oPeSfPr4HTIEA/viewform) by following the [documentation](http://svl.stanford.edu/gibson2/docs/). Please use the `gibson_sim2real` branch instead of the `master` branch.
  ```
  cd GibsonEnvV2
  git checkout gibson_sim2real
  ```
- Step 4: install [our fork of tf-agents](https://github.com/StanfordVL/agents). Please use the `gibson_sim2real` branch instead of the `master` branch.
  ```
  cd agents
  git checkout gibson_sim2real
  pip install tensorflow-gpu==1.15.0
  pip install -e .
  ```
- Step 5: start training!
  ```
  cd agents
  
  # SAC
  ./tf_agents/agents/sac/examples/v1/train_single_env.sh

  # DDPG / PPO
  TBA
  ```
  This will train in one single scene specified by `model_id` in `config_file`.

- Step 6: scale up training!
  ```
  cd agents
  
  # SAC
  ./tf_agents/agents/sac/examples/v1/train_multiple_env.sh

  # DDPG / PPO
  TBA
  ```
  This will train in all the training scenes defined in `GibsonEnvV2/gibson2/data/train.json`. After every `reload_interval` train steps, a new group of scenes will be randomly sampled and reloaded.
  
Feel free to skip Step 4-6 if you want to use other frameworks for training. These are just example starter code for your reference.

Acknowledgments
-------------------
We thank [habitat](https://aihabitat.org/) team for the effort of converging task setup and challenge API. 


References 
-------------------
[1] [Interactive Gibson: A Benchmark for Interactive Navigation in Cluttered Environments](https://ieeexplore.ieee.org/abstract/document/8954627/).  Xia, Fei, William B. Shen, Chengshu Li, Priya Kasimbeg, Micael Tchapmi, Alexander Toshev, Roberto Martín-Martín, and Silvio Savarese. arXiv preprint arXiv:1910.14442 (2019).

[2] [Gibson env: Real-world perception for embodied agents](https://arxiv.org/abs/1808.10654). F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese. In CVPR, 2018

[3] [On evaluation of embodied navigation agents](https://arxiv.org/abs/1807.06757). Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, Amir R. Zamir. arXiv:1807.06757, 2018.
