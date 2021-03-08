package creator

func (ic *irregularChance) divideChances(number int) {
	ic.ChanceFromAllIrregular /= float64(number)
	ic.SwitchedToWhat.divideChances(int(ic.ChanceForType))
	ic.ChanceForType /= float64(ic.counter)
}

func (ic *irregularSlice) divideChances(allCounter int) {
	for i := range *ic {
		(*ic)[i].divideChances(allCounter)
	}
}

func (ig *irregularGlobal) divideChances() {
	ig.Details.divideChances(int(ig.GeneralChance))
	ig.GeneralChance /= float64(ig.usedLocationCounter)
}

func (pcd *chanceSlice) divideChances(number int) {
	if number == 0 {
		return
	}
	numberF := float64(number)
	for i := range *pcd {
		(*pcd)[i].Chance /= numberF
	}
}

func (d *chanceDictDiagSlice) divideChances(number int) {
	numberF := float64(number)
	for i := range *d {
		(*d)[i].Chance /= numberF
	}
}

func (sd *stateSlice) calculatePercentages() {
	for i := range *sd {
		(*sd)[i].Distribution.divideChances((*sd)[i].agentCounter)
	}
}

func (crf *ConfigRandomFormat) calculatePercentages() {
	crf.IrregularLocChance.divideChances()
	crf.AgentTypeDistribution.divideChances(crf.agentCounter)
	crf.LocationTypeDistibution.divideChances(crf.locationCounter)
	crf.PreCondDistibution.divideChances(crf.agentCounter)
	crf.StateDistribution.calculatePercentages()
}
