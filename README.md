# Environment_Shaping_NSE
Python code base for "Mitigating Negative Side Effects via Environment Shaping", published at AAMAS 2021. 
Link to paper: https://arxiv.org/abs/2102.07017

-----------------------------------------------------------------------------------------------------------
The experiments currently support "boxpushing" domain for the single actor, single designer setting, and an AV driving domain for multiple actors, single designer setting. Each environment is described by a map.

Boxpushing domain: The agent is expected to quickly push a box across the room, during which it may dirty the rug or knock over the vase as a negative side effect. 

Driving domain: The agent optimizes travel time between locations. It does not slow down when navigating through potholes, causing bumpy ride for passengers and potentially damaging the car. 

-----------------------------------------------------------------------------------------------------------
File Description:
- env.py: Reads map and solves the problem using Value Iteration
- boxpushing.py: Problem setup for boxpushing domain 
- driving.py: Problem setup for driving domain
- domain_helper.py: Helper functions
- designer.py: Identifies the best modification for the current environment, given agent policy  
- shaping_boxpushing.py: Implements actor-designer coordination and environment shaping for boxpushing domain (single actor setting)
- shaping_multiple_actors.py:Implements actor-designer coordination and environment shaping for driving domain (multiple actor setting)
- feedback_baseline.py: Contains functions to learn from human feedback and update agent policy
- plotresults.py: Generates plots for single actor, single designer setting
- plot_multipleactors_results.py: Generates plots for multiple actors, single designer setting

-----------------------------------------------------------------------------------------------------------

**Code Execution**:
Dependencies: sklearn, numpy

**Single agent shaping boxpushing domain**:
Avoidable NSE: python shaping_boxpushing.py 

Unavoidable NSE: python shaping_boxpushing.py unavoidable

**Shaping for multiple actors**:
python shaping_multiple_actors.py [NUMBER_ACTORS]

Example: python shaping_multiple_actors.py 50

**Feedback_baseline**:
Avoidable NSE: python feedback_baseline.py boxpushing

Generalize feedback results:python feedback_baseline.py boxpushing generalize

Unavoidable NSE: python feedback_baseline.py boxpushing unavoidable

Generalize feedback results:python feedback_baseline.py boxpushing generalize unavoidable

**Generate plots**:
Single actor: python plotresults.py [SHAPING_FILE] [FEEDBACK_FILE] [OP_FILENAME]

Example: python plotresults.py ../results/bp_trials.txt ../results/bp_feedback_baseline_avoidable.txt ../results/bp_feedback_baseline_avoidable_generalize.txt bp

Multiple actors: python plot_multipleactors_results.py [OP_FILENAME]

Example:python plot_multipleactors_results.py driving

