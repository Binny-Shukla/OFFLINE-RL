from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

path = r'C:\LLM & Agents\Trajectory_Transformer\runs\TT\events.out.tfevents.1756336303.Kio.21304.0'

ea = event_accumulator.EventAccumulator(path)
ea.Reload()

scalar = ea.Scalars('Avg TT Loss')

steps = [s.step for s in scalar]
values = [s.value for s in scalar]


plt.plot(steps, values, label = "Avg TT Loss")
#plt.yscale("log")
plt.title("TRAJECTORY TRANSFORMER LOSS")
plt.xlabel("<- EPOCHS ->")
plt.ylabel("<- LOSS ->")
plt.legend()
plt.show()