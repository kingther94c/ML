#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[55]:


df=pd.read_pickle("..//output//records_FrozonGridWorld.pickle")


# In[56]:


fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for algo, df_sub in df[df.algo.apply(lambda x: "PI" in x)].groupby("algo"):
    df_sub = df_sub.set_index('i')[["ER", "dv", "time"]]
    df_sub.time.plot(ax=axs[0], label=algo)
    axs[0].legend()
    axs[0].set_title("Time")
    df_sub.dv.plot(ax=axs[1], label=algo)
    axs[1].legend()
    axs[1].set_title("Delta in value")
    df_sub.ER.plot(ax=axs[2], label=algo)
    axs[2].legend()
    axs[2].set_title("Expected cumulative discounted reward")


# In[57]:


fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for algo, df_sub in df[df.algo.apply(lambda x: "VI" in x)].groupby("algo"):
    df_sub = df_sub.set_index('i')[["ER", "dv", "time"]].iloc[:50]
    df_sub.time.plot(ax=axs[0], label=algo)
    axs[0].legend()
    axs[0].set_title("Time")
    df_sub.dv.plot(ax=axs[1], label=algo)
    axs[1].legend()
    axs[1].set_title("Delta in value")
    df_sub.ER.plot(ax=axs[2], label=algo)
    axs[2].legend()
    axs[2].set_title("Expected cumulative discounted reward")


# In[58]:


fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for algo, df_sub in df[df.algo.apply(lambda x: "Q" in x)].groupby("algo"):
    df_sub = df_sub.set_index('i')[["ER", "dv", "time"]]
    df_sub.time.plot(ax=axs[0], label=algo)
    axs[0].legend()
    axs[0].set_title("Time")
    df_sub.dv.plot(ax=axs[1], label=algo)
    axs[1].legend()
    axs[1].set_title("Delta in avg reward")
    df_sub.ER.plot(ax=axs[2], label=algo)
    axs[2].legend()
    axs[2].set_title("Expected cumulative discounted reward")


# In[60]:


df=pd.read_pickle("..//output//records_SnakesLaddersGame.pickle")


# In[61]:


fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for algo, df_sub in df[df.algo.apply(lambda x: "PI" in x)].groupby("algo"):
    df_sub = df_sub.set_index('i')[["ER", "dv", "time"]].iloc[:30]
    df_sub.time.plot(ax=axs[0], label=algo)
    axs[0].legend()
    axs[0].set_title("Time")
    df_sub.dv.plot(ax=axs[1], label=algo)
    axs[1].legend()
    axs[1].set_title("Delta in value")
    df_sub.ER.plot(ax=axs[2], label=algo)
    axs[2].legend()
    axs[2].set_title("Expected cumulative discounted reward")


# In[62]:


fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for algo, df_sub in df[df.algo.apply(lambda x: "VI" in x)].groupby("algo"):
    df_sub = df_sub.set_index('i')[["ER", "dv", "time"]].iloc[:30]
    df_sub.time.plot(ax=axs[0], label=algo)
    axs[0].legend()
    axs[0].set_title("Time")
    df_sub.dv.plot(ax=axs[1], label=algo)
    axs[1].legend()
    axs[1].set_title("Delta in value")
    df_sub.ER.plot(ax=axs[2], label=algo)
    axs[2].legend()
    axs[2].set_title("Expected cumulative discounted reward")


# In[63]:


fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for algo, df_sub in df[df.algo.apply(lambda x: "Q" in x)].groupby("algo"):
    df_sub = df_sub.set_index('i')[["ER", "dv", "time"]]
    df_sub.time.plot(ax=axs[0], label=algo)
    axs[0].legend()
    axs[0].set_title("Time")
    df_sub.dv.plot(ax=axs[1], label=algo)
    axs[1].legend()
    axs[1].set_title("Delta in avg reward")
    df_sub.ER.plot(ax=axs[2], label=algo)
    axs[2].legend()
    axs[2].set_title("Expected cumulative discounted reward")


# In[ ]:




