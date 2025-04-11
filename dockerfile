FROM htorch/hasktorch-jupyter:latest-cpu

RUN curl -sSL https://get.haskellstack.org/ | sh

RUN stack upgrade
RUN stack --version
RUN stack setup

RUN mkdir libraries
RUN mv ./hasktorch ./inline-c ./dist-newstyle ./libraries/

WORKDIR /home/ubuntu/libraries
RUN git clone https://github.com/DaisukeBekki/hasktorch-tools.git
RUN git clone https://github.com/DaisukeBekki/nlp-tools.git

WORKDIR /home/ubuntu/libraries/hasktorch-tools
# for stack run
RUN sed -i -e "s|/Users/evanrimonteil/Documents/IUT/Stage/hasktorch|/home/ubuntu/libraries/hasktorch|g" /home/ubuntu/libraries/hasktorch-tools/stack.yaml

WORKDIR /home/ubuntu
