# Setting up Cloudlab

You can do this assignment using CloudLab. CloudLab is a research facility that provides bare-metal access and control over a substantial set of computing, storage, and networking resources. If you haven’t worked in CloudLab before, you need to register a CloudLab account.

This write-up walks you through the CloudLab registration process and shows you how to start an experiment in CloudLab.

Most importantly, it introduces our policies on using CloudLab for this project.

## Register a CloudLab Account

To register an account, please visit [http://cloudlab.us](http://cloudlab.us) and create an account using your Princeton email address as login. Note that an SSH public key is required to access the nodes CloudLab assigns to you; if you are unfamiliar with creating and using ssh keypairs, we recommend taking a look at the first few steps in [GitHub’s guide to generating SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh). (Obviously, the steps about how to upload the keypair into GitHub don’t apply to CloudLab.) Click on `Join Existing Project` and enter `cos598dsp24` as the project name. Then click on `Submit Request`. The project leader will approve your request. If you already have a CloudLab account, simply request to join the `cos598dsp24` project.

## Start an Experiment

To start a new experiment, go to your CloudLab dashboard and click on the `Experiments` tab in the upper left corner, then select `Start Experiment`. This will lead to the profile selection panel. Click on `Change Profile`, and select a profile from the list. For example, if you choose the `cos598d-distributed-ml` profile in the `cos598dsp24` project, you will be able to launch 4 CPU nodes. Select the profile and click on Next to move to the next panel. Here you should name your experiment with `NetID-ExperimentName`. The purpose of doing this is to prevent everyone from picking random names and ending up confusing each other since everyone in the `cos598dsp24` project can see a full list of experiments created. You also need to specify from which cluster you want to start your experiment. Each cluster has different hardware. For more information on the hardware CloudLab provides, please refer to [this doc](https://docs.cloudlab.us/hardware.html). Once you select the cluster you can instantiate the experiment. Once the experiment is ready you will see the ssh login command. Try to log in to the machine and check for the number of CPU cores available and memory available on the node using `htop` (You might need to install htop first by running `sudo apt-get update ; sudo apt-get install htop`).

## Policies on Using CloudLab Resources

The nodes you receive from CloudLab are real hardware machines sitting in different clusters. Therefore, we ask you not to hold the nodes for too long. CloudLab gives users 16 hours to start with, and users can extend it for a longer time. Manage your time efficiently and only hold onto those nodes when you are working on the assignment. You should use a private git repository to manage your code, and you must terminate the nodes when you are not using them. If you do have a need to extend the nodes, do not extend them by more than 1 day. We will terminate any cluster running for more than 48 hours.
As a member of the `cos598dsp24` project, you have permission to access another member’s private user space. Stick to your own space and do not access others’ to peek at/copy/use their code, or intentionally/unintentionally overwrite files in others’ workspaces. For more information related to this, please refer to [https://odoc.princeton.edu/learning-curriculum/academic-integrity](https://odoc.princeton.edu/learning-curriculum/academic-integrity).
