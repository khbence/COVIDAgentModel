package creator

type (
	chanceDict struct {
		Value  string  `json:"value"`
		Chance float64 `json:"chance"`
	}

	chanceDictDiag struct {
		chanceDict
		DiagChance float64 `json:"diagnosedChance"`
	}

	irregularChance struct {
		Value                  string      `json:"value"`
		ChanceForType          float64     `json:"chanceForType"`
		ChanceFromAllIrregular float64     `json:"chanceFromAllIrregular"`
		SwitchedToWhat         chanceSlice `json:"switchedToWhat"`
		counter                int
	}

	irregularGlobal struct {
		GeneralChance       float64        `json:"generalChance"`
		Details             irregularSlice `json:"detailsOfChances"`
		usedLocationCounter int
	}

	chanceSlice         []chanceDict
	stateSlice          []stateData
	ageIntervalSlice    []AgeInterval
	chanceDictDiagSlice []chanceDictDiag
	irregularSlice      []irregularChance

	stateData struct {
		AgeStart     int                 `json:"ageStart"`
		AgeEnd       int                 `json:"ageEnd"`
		Distribution chanceDictDiagSlice `json:"distribution"`
		agentCounter int
	}

	// ConfigRandomFormat stores the entire file
	ConfigRandomFormat struct {
		IrregularLocChance      irregularGlobal `json:"irregulalLocationChance"`
		LocationTypeDistibution chanceSlice     `json:"locationTypeDistibution"`
		PreCondDistibution      chanceSlice     `json:"preCondDistibution"`
		StateDistribution       stateSlice      `json:"stateDistibution"`
		AgentTypeDistribution   chanceSlice     `json:"agentTypeDistribution"`
		agentCounter            int
		locationCounter         int
		locationMap             map[string]string
		signalLocation          chan bool
	}

	// AgeInterval is used to define what are the intervals to get the state distributions
	AgeInterval struct {
		Begin int
		End   int
	}
)
