from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

logdir = r'C:\LLM & Agents\Decision_Transformer\.runs\DT\events.out.tfevents.1756161129.Kio.8636.0'

ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()


scalar = ea.Scalars('DT_LOSS')

steps = [s.step for s in scalar]
values = [s.value for s in scalar]

plt.plot(steps, values, label = "DT_LOSS")
plt.yscale("log")
plt.title("DECISION TRANSFORMER LOSS")
plt.xlabel("<- EPOCHS ->")
plt.ylabel("<- LOSS ->")
plt.legend()
plt.show()