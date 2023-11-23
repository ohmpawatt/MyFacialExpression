# qMDP
## Install
1. 安装项目依赖
	```
	pip install -r requirements.txt
	```
2. 在gym\envs\__init__.py中添加

	```
	register(
    	id ="quantumGridWorld",
    	entry_point="gym.envs.classic_control:GridEnv",
    	reward_threshold=100.0,
    	max_episode_steps=200,
	)
	```
3. 将environment.py复制到\gym\envs\classic_control
4. 在\gym\envs\classic_control\__init__.py中添加

	```
	from gym.envs.classic_control.environment import GridEnv
	```







