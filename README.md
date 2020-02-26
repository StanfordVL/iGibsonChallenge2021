Sim2Real Challenge with Gibson @ CVPR 2020
=============================================

This repository contains starter code for sim2real challenge with Gibson brought to you by [Stanford VL](http://svl.stanford.edu) and [Robotics @ Google](https://research.google/teams/brain/robotics/). 
For an overview of the challenge, visit [the challenge website](http://svl.stanford.edu/gibson2/challenge.html).

Challenge Scenarios
----------------------------
The first Gibson Sim2Real Challenge is composed of three navigation scenarios that represent important skills for autonomous visual navigation: 

- PointNav scenario in clean environments: the goal in this scenario is for an agent to successfully navigate to a given point location based on visual information (RGB+D images). In this scenario, the agent is not allowed to collide with the environment. This scenario will evaluate the sim2real transference of the most basic capability of a navigating agent. We will evaluate performance in this scenario using Success weighted by Path Length (SPL) [3].

- PointNav scenario with interactive objects: in this scenario the agent is allowed (even encouraged) to collide and interact with the environment in order to push obstacles away. But careful! Some of the obstacles are not movable. This scenario evaluates agents in Interactive Navigation tasks [1], navigation problems that considers interactions with the environment. We will use Interactive Navigation Score (INS) [1] to evaluate performance of agents in this scenario.

- PointNav scenario among dynamic agents: the goal in this scenario is to navigate to the given point location avoiding collisions with a dynamic agent that follows unknown navigating patterns. Reasoning, predicting and avoiding other moving agents is challenging, and we will measure how well existing solutions perform in this conditions. As with PointNav scenarios in clean environments, no collisions are allowed in this scenario. We will use again SPL to evaluate the performance of the controlled agent.
All submissions to our challenge will be evaluated in all three scenarios. The ranking will depend on the performance on the three scenarios but we will provide insights about the performance in each of them.

Challenge Dataset
----------------------------

We used navigation episodes Habitat created as the main dataset for the challenge. In addition, we scanned a 
new house named "Castro" and use part of it for training. The evaluation will be in Castro in both sim and real.

- Training scenes: 72 Gibson Scenes + Castro
- Dev scenes: Castro(Sim), Castro(Real)
- Evaluation scenes: CastroUnseen(Sim), CastroUnseen(Real)


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
  git clone https://github.com/StanfordVL/GibsonSim2RealChallenge.git
  cd GibsonSim2RealChallenge
  ```

  Three example agents are provided in `simple_agent.py` and `rl_agent.py`: `RandomAgent`, `ForwardOnlyAgent`, and `SACAgent`.
  
  Here is the `RandomAgent` defined in `simple_agent.py`.
  ```python3
  ACTION_DIM = 2
  LINEAR_VEL_DIM = 0
  ANGULAR_VEL_DIM = 1


  class RandomAgent:
      def __init__(self):
          pass

      def reset(self):
          pass

      def act(self, observations):
          action = np.random.uniform(low=-1, high=1, size=(ACTION_DIM,))
          return action
  ```
  
  Please, implement your own agent and instantiate it from `agent.py`.

- Step 2: Install nvidia-docker2, following the guide: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0). 

- Step 3: Modify the provided Dockerfile to accommodate any dependencies. A minimal Dockerfile is shown below.
  ```Dockerfile
  FROM gibsonchallenge/gibsonv2:latest
  ENV PATH /miniconda/envs/gibson/bin:$PATH

  ADD agent.py /agent.py
  ADD submission.sh /submission.sh
  WORKDIR /
  ```

  Then build your docker container with `docker build . -t my_submission` , where `my_submission` is the docker image name you want to use.

- Step 4: 

  Download challenge data from [here](https://docs.google.com/forms/d/e/1FAIpQLSen7LZXKVl_HuiePaFzG_0Boo6V3J5lJgzt3oPeSfPr4HTIEA/viewform) and decompress in `GibsonSim2RealChallenge/gibson-challenge-data`. The file you need to download is called `gibson-challenge-data.tar.gz`.
  
  Please, change the permissions of the directory and subdirectories:
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
Follow instructions in the submit tab of the EvalAI challenge page to submit your docker image. Note that you will need a version of EvalAI >= 1.2.3. Here we reproduce part of those instructions for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.2.3"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push my_submission:latest --phase <phase-name>
```

The valid challenge phases are: `sim2real-{minival-535, challenge-sim-535,  challenge-real-535}`.

Our Sim2Real Challenge consists of three phases:

- Phase 0: Sanity Check (`minival`): 
The purpose of this phase is mainly for sanity checking and make sure the policy 
can be successfully submitted and evaluated.
Participant can submit any policy, even a trivial policy, to our evaluation server to verify their entire pipeline is correct. 
Each team is allowed maximum of 30 submission per day for this phase. 
We will block and disqualify teams that spam our servers.

- Phase 1, Simulation phase (`challenge-sim`): In this phase participants will develop their solutions in our simulator, Interactive Gibson. They will get access to all Gibson 3D reconstructed scenes (572 total, 72 high quality ones, which we recommend for training) and an additional 3D reconstructed scene called Castro that contains part of real world apartment we will use in Phase 2. We will keep the other part of the apartment, which we call CastroUnseen, to perform the evaluation. Participants can submit their solutions at any time through the EvalAI portal (link coming soon). At the end of the simulation challenge period (May 15), the best ten solutions will pass to the second phase, the real world phase. As part of our collaboration with Facebook, the top five teams from the Habitat Challenge will also take part in the phase 2 of our challenge and will test their solutions in real world.

- Phase 2, Real world phase (`challenge-real`): The qualified teams will receive 30 min/day to evaluate their policies on our real world robotic platform. The runs will be recorded and the videos will be provided to the teams for debugging. They will also receive a record of the states, measurements, and actions taken by the real world agent, as well as their score. The last two days (31st of May and 1st of June) are the days of the challenge and the teams will be ultimately ranked based on their scores. At the end of these two days we will announce the winner of the first Gibson Sim2Real Challenge!

- Phase 3, Demo phase: To increase visibility, the best three entries of our challenge will have the opportunity to showcase their solutions live during CVPR20! We will connect directly from Seattle the 15th of June and video stream a run of each solutions, highlighting their strengths and characteristics. This will provide an opportunity for the teams to explain their solution to the CVPR audience. This phase is not included in our EvalAI setup.


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
