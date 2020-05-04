# General info
If a value is not relevant then for unsigned integer values it should be -1, for strings an empty string.
If some data cannot be found and hard to estimate we should talk about it, but the aformentioned cases should have a trivial behaviour in the simulation
ID is unique for the given instance if it's not specified here.
## locations.json
* id: unique for every locations (this will be referenced in the agents schedule)
* type: is the reference to the locationTypes.json's id field.
* state: can be ON/OFF, I'm not sure if we'll need this, because I don't know how we'll change it's state during the simulation
## agentTypes.json
* id: unique id
* schedule/id: unique inside the type, but not throughout the types
* schedule/type: reference to locationTypes' id
* scheduleTypic: it's meant to reduce the repetitive schedules, it's id is the same as schedule/id, the scheduleId is reference to the commonSchedules's id on the top of the file
## agents.json
* typeId: reference to the agentTypes's id
* locations/typeID: reference to agentTypes.json/schedule/id
* locations/locID: reference to location.json/id