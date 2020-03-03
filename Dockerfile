FROM gibsonchallenge/gibsonv2:latest
ENV PATH /miniconda/envs/gibson/bin:$PATH

ADD agent.py /agent.py
ADD simple_agent.py /simple_agent.py
ADD rl_agent.py /rl_agent.py

ADD submission.sh /submission.sh
WORKDIR /
