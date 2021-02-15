
## Input options
Details about the format and contents in input json files are discussed [here](inputFiles/README.md)
|short|long|details
|--- | --- | ---
|  -w| --weeks                   | Length of simulation in weeks (default: 12)
|  -t| --deltat                  | Length of timestep in minutes (default: 10)
|  -n| --numagents               | Number of agents (default: -1)
|  -N| --numlocs                 | Number of dummy locations (default: -1)
|  -P| --progression             | Path to the config file for the progression matrices. (default: ../inputFiles/progressions/transition_config.json)
|  -a| --agents                  | Agents file, for all human being in the experiment. (default: ../inputFiles/agents.json)
|  -A| --agentTypes              | List and schedule of all type fo agents. (default: ../inputFiles/agentTypes.json)
|  -l| --locations               | List of all locations in the simulation. (default: ../inputFiles/locations.json)
|  -L| --locationTypes           | List of all type of locations (default: ../inputFiles/locationTypes.json)
|  -p| --parameters              | List of all general parameters for the simulation except the progression data. (default: ../inputFiles/parameters.json)
|  -c| --configRandom            | Config file for random initialization. (default: ../inputFiles/configRandom.json)
|    | --closures                | List of closure rules. (default: ../inputFiles/closureRules.json)
|  -r| --randomStates            | Change the states from the agents file with the configRandom file's stateDistribution.
|    | --outAgentStat            | name of the agent stat output file, if not set there will be no print (default: "")
|    | --diags                   | level of diagnositcs to print (default: 0)
|    | --otherDisease            | Enable (1) or disable (0) non-COVID related hospitalization and sudden death  (default: 1)
|    | --mutationMultiplier      | infectiousness multiplier for mutated virus (default: 1.0)
|  -k| --infectionCoefficient    | Infection: >0 :infectiousness coefficient (default: 0.000374395)
|    | --dumpLocationInfections  | Dump per-location statistics every N timestep  (default: 0)
|    | --dumpLocationInfectiousList | Dump per-location list of infectious people (default: "")
|    | --trace                   | Trace movements of agent (default: 4294967295)
|    | --quarantinePolicy        | Quarantine policy: 0 - None, 1 - Agent only, 2 - Agent and household, 3 - + classroom/work, 4 - + school (default: 3)
|    | --quarantineLength        | Length of quarantine in days (default: 10)
|    | --testingProbabilities    | Testing probabilities for random, if someone else was diagnosed at home/work/school, and random for hospital workers: comma-delimited string random,home,work,school,hospital,nurseryHome (default: 0.0001,0.02,0.001,0.001,0.01,0.1)
|    | --testingRepeatDelay      | Minimum number of days between taking tests (default: 5)
|    | --testingMethod           | default method for testing. Can be PCR (default) on antigen. Accuracies are provided in progression json input (default: PCR)
|    | --enableClosures          | Enable(1)/disable(0) closure rules defined in closureRules.json (default: 1)
|    | --disableTourists         | enable or disable tourists (default: 1)
|    | --immunizationStart       | number of days into simulation when immunization starts (default: 0)
|    | --immunizationsPerDay     | number of immunizations per day (default: 0)
|    | --immunizationOrder       | Order of immunization (starting at 1, 0 to skip) for agents in different categories health workers, nursery home worker/resident, 60+, 18-60 with underlying condition, essential worker, 18+ (default: 1,2,3,4,5,6)
|  -h| --help                    | Print usage
|    | --version| 
