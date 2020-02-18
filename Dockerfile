FROM gibsonchallenge/gibsonv2:latest
ENV PATH /miniconda/envs/gibson/bin:$PATH

ADD agent.py /agent.py
ADD submission.sh /submission.sh
WORKDIR /