package creator

func (pcd *chanceSlice) increment(key string) {
	found := false
	for i, value := range *pcd {
		if value.Value == key {
			(*pcd)[i].Chance += 1.0
			found = true
			break
		}
	}
	if !found {
		*pcd = append(*pcd, chanceDict{Value: key, Chance: 1.0})
	}
}

func (d *chanceDictDiagSlice) increment(state string) {
	found := false
	for i, value := range *d {
		if value.Value == state {
			(*d)[i].Chance++
			found = true
			break
		}
	}
	if !found {
		*d = append(*d, chanceDictDiag{
			chanceDict: chanceDict{
				Value:  state,
				Chance: 1.0,
			},
			DiagChance: 0.0,
		})
	}
}

func (ic *irregularSlice) increment(expected, real string) bool {
	found := false
	irregular := false
	for i, value := range *ic {
		if value.Value == expected {
			found = true
			current := &(*ic)[i]
			current.counter++
			if expected != real {
				current.ChanceForType++
				current.ChanceFromAllIrregular++
				current.SwitchedToWhat.increment(real)
				irregular = true
			}
			break
		}
	}
	if !found {
		newElement := irregularChance{
			Value:   expected,
			counter: 1,
		}
		if irregular {
			newElement.ChanceForType++
			newElement.ChanceFromAllIrregular++
			newElement.SwitchedToWhat.increment(real)
		}
		*ic = append(*ic, newElement)
	}
	return irregular
}

func (sd *stateSlice) increment(age int, state string) {
	for i, value := range *sd {
		if value.AgeStart < age && age < value.AgeEnd {
			(*sd)[i].agentCounter++
			(*sd)[i].Distribution.increment(state)
		}
	}
}
