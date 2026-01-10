# DeliveryBench: Can Agents Earn Profit in Real World?

</p> <p align="center"> 📄 <a href="#">Paper</a> &nbsp; | &nbsp; 🏠 <a href="#">Website</a> &nbsp; | &nbsp; 🌍 <a href="#">SimWorld-Env</a> </p> <p align="center"> 

<p align="center">
  <a href="LINK_MAO">Lingjun Mao<sup>1</sup></a>, 
  <a href="LINK_REN">Jiawei Ren<sup>1</sup></a>, 
  <a href="LINK_ZHOU">Kun Zhou<sup>1</sup></a>, 
  <a href="LINK_CHEN">Jixuan Chen<sup>1</sup></a>, 
  <a href="LINK_MA">Ziqiao Ma<sup>2</sup></a>, 
  <a href="LINK_QIN">Lianhui Qin<sup>1</sup></a>
</p>

<p align="center">
  <sup>1</sup> University of California, San Diego &nbsp; &nbsp; 
  <sup>2</sup> University of Michigan
</p>

<p align="center">
  <img src="https://www.dropbox.com/scl/fi/5o17v7mct4ew2c6n2f4be/deliverybench.png?rlkey=3dhazn7ewf1fna6krhg00h4ks&st=eb2whx62&raw=1" alt="image" />
</p>

## 💡 Introduction
**DeliveryBench** is a city-scale embodied benchmark that evaluates whether VLM agents can earn profit under realistic, long-horizon constraints. Agents act as autonomous couriers in 3D cities, accepting and completing delivery orders across multiple in-game hours. They must manage resources (e.g., stamina, e-scooter battery), adapt to changing conditions, and balance efficiency, timing, and cost. When multiple agents coexist, they also face social dynamics such as competition and collaboration. By jointly modeling economic, physical, and social dynamics within a unified embodied environment, DeliveryBench provides a realistic, action-driven setting to test whether VLM-based agents can plan and act strategically to improve financial outcomes.

## 🔍 Key Features

<p align="center">
  <img src="https://www.dropbox.com/scl/fi/bku8d5lw9qh90mi8mq7bp/comparison.png?rlkey=j99bpymop97gatmsbiw861s2c&st=y6stsh2j&raw=1" alt="image" />
</p>


Compared with prior embodied benchmarks, DeliveryBench supports long-horizon tasks (several in-game hours; typically > 100 action steps) with multi-dimensional real-world constraints, covering:

**⏳ Time Constraints**: Tasks have deadlines and time windows that determine when they can be performed.
Agents must schedule actions to avoid late deliveries and make efficient use of limited working time.

**🗺️ Spatial Constraints**: Some actions are only valid at specific locations, so agents must navigate 3D cities and visit the right POIs in the right order (e.g., restaurants, charging stations).

**🔋 Resource Constraints**: Agents must manage consumables such as stamina, vehicle battery, and cash to stay operational, sometimes transforming one resource into another (e.g., buying an energy drink to restore stamina).

**⚙️ Physical Constraints**: Environmental dynamics (e.g., temperature, motion, collisions) affect food quality, requiring agents to consider item fragility and perishability in route planning.

**💵 Economic Constraints**: Agents earn income but also pay operational costs (e.g., recharging, renting, buying supplies), forcing them to balance short-term expenses against long-term profit.

**🤝 Social Constraints**: In multi-agent settings, couriers collaborate and compete for limited opportunities (e.g., high-value orders, charging spots), shaping both strategy and outcomes.

## 🚀 Setup
### Project Structure
```bash
evaluation/             # Evaluation and analysis utilities

maps/                   # Test city maps and map configs used in benchmark tasks

simworld/               # Core simulation backend (Python API for the UE-based SimWorld engine);
                        # see the SimWorld repo for detailed documentation

vlm_delivery/           # VLM-based delivery agent implementation and runtime
    actions/            # Concrete agent actions (e.g., ACCEPT_ORDER, MOVE_TO, BUY)
    base/               # Shared base classes (e.g., timers, type definitions)
    communicator/       # # Interface between Python and the Unreal Engine environment (UnrealCV API)
    entities/           # Entity classes (e.g., DeliveryMan, Order, vehicles)
    gameplay/           # Runtime logic such as run_recorders, prompt construction
    input/              # Task and environment configuration (food types, agent count, etc.)
    vlm/                # Core VLM wrapper classes and model interfaces
    map/                # Map abstractions, waypoint systems, and visualization utilities used by the agent
    scripts/            # Test scripts to quickly run DeliveryBench
    utils/              # Helper utilities and common functions

.gitignore
README.md
```

### Installation
+ Python Client
Make sure to use Python 3.10 or later.
```bash
git clone https://github.com/mao1207/DeliveryBench.git
cd DeliveryBench
conda create -n deliverybench python=3.10
conda activate deliverybench
pip install -e .
```

+ UE server
Download the SimWorld server executable from S3:

    + Windows. Download the [SimWorld Windows 64-bit Server (v0.1.0)](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Win64-v0_1_0-Foundation.zip) and unzip it.
    + Linux. Download the [SimWorld Linux 64-bit Server (v0.1.0)](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Linux-v0_1_0-Foundation.zip) and unzip it.

## ▶️ Quick Start

### 1. Launch the SimWorld server

Navigate to the directory containing the SimWorld executable and run:

```bash
gym_citynav.exe <map_name>
```

Supported `map_name` options include:

```
small-city-11, small-city-13, small-city-15,
medium-city-18, medium-city-20, medium-city-22,
medium-city-28, medium-city-30,
large-city-26
```

By default, the UE environment runs on **port 9000**.
You can modify the port in:

```
gym_citynav/Binaries/Linux/unrealcv.ini
```

---

### 2. Configure experiments

Experimental settings are specified in:

```
vlm_delivery/input/config.json
```

Make sure that:

* the **map name** matches the one used for `gym_citynav.exe`
* the **port number** is consistent with the UE server

In addition, you can specify:

* **trajectory_output_dir**:
  Directory for logging every step of the agent’s trajectory.
  Each step includes the agent’s inputs and outputs, such as the observed global map, first-person view, and the selected action.

* **life_cycle.export_path**:
  Path for exporting episode-level results after a run completes (e.g., when `life_cycle.duration_hours` is reached).
  The exported JSON file contains detailed records of the run, which can be used for post-hoc statistics and analysis.

Other parameters relate to the delivery task mechanics, such as movement speed and energy consumption for different transportation modes. These can be modified as needed, although we recommend using the default settings to match the experimental setup in our paper.

---

### 3. Configure models

The VLM model is defined in:

```
vlm_delivery/input/model.json
```

You can directly swap in models supported by **OpenRouter** or **OpenAI** (e.g., gpt-4o, gpt-4.1, llama-3.1, etc.).

Just replace the model name and corresponding API key fields.

---

### 4. Run the DeliveryBench evaluation

Open the quick-start notebook:

```
vlm_delivery/scripts/run_deliverybench.ipynb
```

This notebook will:

* connect to the UE server
* spawn courier agents
* run delivery episodes
* log and visualize results

---

### 5. Evaluating Custom Models

We also support plugging in **local VLMs**.
As a reference, we provide a minimal implementation of **LLaVA-OneVision** in:

```
vlm_delivery/vlm/base_model.py
```

You can adapt this file to wrap your own local model (e.g., by following the same `forward` / `generate` interface and image/text preprocessing pipeline).

The lightweight agentic workflow (including chain-of-thought reasoning and future planning) is implemented through:

* `vlm_delivery/gameplay/prompt.py` — prompt templates for actions, CoT, and future plans
* `vlm_delivery/gameplay/action_space.py` — parsing model outputs into structured actions
* `vlm_delivery/utils/vlm_prompt.py` — runtime prompt assembly (e.g., feeding the previous plan, observations, or action history back into the model)

You are free to extend this workflow with additional modules, such as:

* **memory modules** (e.g., episodic or long-term memory over past orders and routes)
* **reflection / self-correction loops** (e.g., asking the model to critique or refine its own plan)
* **tool-use modules** (e.g., calling external routing APIs or heuristic planners before acting)

### 6. Analyze results

After runs finish and JSON result files are exported (via `life_cycle.export_path`), you can aggregate them into CSV summaries using:

```bash
python vlm_delivery/evaluation/agent_performance_analysis.py \
  /path/to/result_json_folder \
  -o /path/to/output_dir
````

* `/path/to/result_json_folder` should point to a directory containing one or more JSON result files.
* The script will automatically load **all JSON files in the folder**, compute aggregate statistics (e.g., per-model averages), and write the CSV reports into `/path/to/output_dir`.

## 🧑‍💻For Contributors
### Precommit Setup

We use Google docstring format for our docstrings and the pre-commit library to check our code. To install pre-commit, run the following command:

```bash
conda install pre-commit  # or pip install pre-commit
pre-commit install
```

The pre-commit hooks will run automatically when you try to commit changes to the repository.

### Commit Message Guidelines
All commit messages should be clear, concise, and follow this format:
```
<type>: <short summary>

[optional body explaining the change]
```
Recommended types:
+ feat: A new feature
+ fix: A bug fix
+ docs: Documentation changes
+ refactor: Code restructuring without behavior changes
+ style: Code style changes (formatting, linting)
+ test: Adding or updating tests
+ chore: Non-code changes (e.g., updating dependencies)

Example:
```
feat: add user login API
```

### Issue Guidelines
+ Use clear titles starting with [Bug] or [Feature].
+ Describe the problem or request clearly.
+ Include steps to reproduce (for bugs), expected behavior, and screenshots if possible.
+ Mention your environment (OS, browser/runtime, version, etc.).

### Pull Request Guidelines
+ Fork the repo and create a new branch (e.g., feature/your-feature, fix/bug-name).
+ Keep PRs focused: one feature or fix per PR.
+ Follow the project’s coding style and naming conventions.
+ Test your changes before submitting.
+ Link related issues using Fixes #issue-number if applicable.
+ Add comments or documentation if needed.

We appreciate clean, well-described contributions! 🚀