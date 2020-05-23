@echo OFF

rem for %%P in (Hopper-v2 , Ant-v2 , HalfCheetah-v2 , Humanoid-v2 , Reacher-v2 , Walker2d-v2) do (
for %%P in (Ant-v2) do (
rem echo %%P
python run_expert.py experts/%%P.pkl %%P --render --num_rollouts=10
)