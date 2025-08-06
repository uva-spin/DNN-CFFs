# DNN-CFFs

The goal of this study is to find a method to accurately predict the values of functions called Compton Form Factors (CFFs) using data from Deep-Virtual Compton Scattering (DVCS) experiments. The DVCS process is $ep\to e' p \gamma$.  Where the interaction between the incoming electron and the target nucleon is mediated through a virtual photon. DVCS data is unique in that it is a deep process, meaning it probes deep inside the target hadron, and it has an additional axis of information from the outgoing real photon.  The process can then provide more spatial information than DIS.

Each experiment, or data set, consists of measurements of the total cross-section ($F$) as an angle ($\phi$) is changed. Since $\phi$ is an angle measurement, its range is from 0° to 360°. At each angle that is used.  (see Figure:  <img width="600" alt=" $F$ vs $\phi$ " src="https://github.com/uva-spin/DNN-CFFs/files/12543136/FvsPhi_Sample.pdf">).
The experiment configuration and the organization of the data results in a limited number of angle bins, so angles of close proximity end up being in the same angle bin with the same total cross-section value for '$F$'. The ‘F’ column of the data file and the standard deviation are recorded in the ‘errF’ column.  Here the 'errF' represents the experimental uncertainty which we assume a Gaussian distribution for. Table shows a sample data structure.


| #Set | index | $k$ | $QQ$ | $x_b$ | $t$ | $\phi$ | $F$ | $errF$ | $F_1$ | $F_2$ |	$dvcs$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | - | - | - | - | - | - | - |

Different experiments experimental goals and configurations lead to different values of the kinematic variables $k$, $Q^2$, $x_b$, and $t$. Each set of values of these variables is called a kinematic setting.  These variables, other than $k$, the incoming beam energy, also have bins, so each numerical value for each of these kinematic variables is an average of many DVCS processes acquired within each bin.  The extraction of the DVCS cross-section and CFFs at any $Q^2$, $x_b$, and $t$ is our ultimate goal.  The CFFs are a non-linear multidimensional function of these three kinematic values.

The total cross-section $F$ is a combination of the DVCS cross-section, the interference, and the Bethe-Heitler and is a known function of the CFFs, the kinematic variables, as well as $\phi$, $k$, $F_1$, $F_2$, which are all known and their values are provided in each data file.

To perform a local fit using a DNN you must construct a network that takes in the kinematic variable $Q^2$, $x_b$, and $t$ and outputs the three CFFs and the DVCS cross-section.  To determine if the CFFs and DVCS cross section are good you must define a loss function based on the total cross section '$F$'.  The loss function uses '$F$' to determine if the steps taken in the iteration process are good or not via the optimizer.  '$F$' is used to calculate the gradient in the SGD or Adam of whatever optimizer is used.  The resulting error from the loss function is then backpropagated through the network to update the weights and biases to better fit the true data provided.

The values of F1, F2, and dvcs are known, so the only unknowns left are the Compton Form Factors. Our goal in this experiment is to determine the values of the CFFs using the other values. We are focusing on three Compton Form Factors: $ReH$, $ReE$, and $Re\tilde{H}$.

There are two kinds of fits that are used to determine the Compton Form Factors: local fits and global fits. A local fit’s end goal is to find the numerical value of one of the CFFs in a certain kinematic setting. A global fit’s end goal is to find the equation that relates the values of the kinematic variables to the value of one of the Compton Form Factors. A local fit is a guess of the value of a CFF in a certain kinematic setting, while a global fit is a guess of the equation for a CFF that can be used to calculate the value of the CFF in any kinematic setting.
One important aspect of this exploration is correctly propagating the error from the cross- section to the Compton Form Factors. We are not given exact values for F, so we cannot possibly get exact values for the CFFs. Every output from our models is given as a guess and an error range. Our goal is to have the guesses be as close as possible to the true values of the CFFs, but it is also important that the error ranges consistently enclose the true values.
The most popular way of propagating error is to use the replica method. In this method, the F value from the data file is not inputted directly into the model. Instead, we use several values that are sampled from a Gaussian distribution with mean equal to F and standard deviation equal to errF. By using input values that represent the distribution of the cross-section, we hope to be able to get outputs that represent the distributions of the Compton Form Factors (meaning that the true value of the CFF should lie in the error bars produced by the model).

Right now, we are using pseudo-data where we know the true values of the CFFs instead of real experimental data where we don’t. This way, we can compare the CFF guesses to their true values and have a good idea of a model’s performance. Once we develop a method that can consistently and accurately predict the CFFs, we will move on to experimental data.

## Sample Codes to Get Started

You can find a folder called "Sample_Codes_to_Get_Started" in the main repository. Instructions to test those codes are given in the ReadMe file in that directory.


## Prerequisites

To run the code and reproduce the environment, you need to have Anaconda installed.

## Installing Anaconda

Follow the steps below to install Anaconda:

1. **Download Anaconda**
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```

2. **Run the Installation Script**
```bash
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

3. **Follow the On-Screen Prompts**: 
Accept the license, and choose the installation location (or accept the default).

4. **Close and Reopen Your Terminal**: 
This ensures that Anaconda is initialized and added to your `PATH`.

5. **Verify Installation**:
To confirm that Anaconda was installed successfully, run
```bash
conda --version
```

## Setting Up the Environment

Once Anaconda is installed, you can set up the environment using the provided `env.yml` file.

1. **Clone the Repository**:
If you haven't already, clone this repository to your local machine:
git clone https://github.com/uva-spin/DNN-CFFs.git
```bash
cd DNN-CFFs
```

2. **Create the Conda Environment** (OUTDATED):
Use the `env.yml` file to create a new Conda environment:
```bash
conda env create -f env.yml
```

2. **AEGIS USERS ONLY: Create the Conda Environment**:
Use the `env_aegis.yml` file to create an environment for aegis. Keep in mind that numpy, pandas, etc are not included, so just download an appropriate version. ALSO, PLEASE CHANGE THE NAME OF THE ENVIRONMENT AS TO NOT CREATE DUPLICATE ENVIRONMENTS ON AEGIS:
```bash
conda env create -f env_aegis.yml --name aegis_env
```

3. **Activate the Environment**:
```bash
conda activate aegis_env
```

Now, you're all set. You can run the project code within this environment. each time you wish
to run a .py file in rivanna run conda activate env first
