package creator

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"statistics_creator/utils"
	"strconv"
	"sync"
)

type (
	chanceDict struct {
		Value  string  `json:"value"`
		Chance float64 `json:"chance"`
	}

	chanceDictDiag struct {
		Value      string  `json:"value"`
		Chance     float64 `json:"chance"`
		DiagChance float64 `json:"diagnosedChance"`
	}

	chanceSlice         []chanceDict
	stateSlice          []stateData
	ageIntervalSlice    []AgeInterval
	chanceDictDiagSlice []chanceDictDiag

	stateData struct {
		AgeStart     int                 `json:"ageStart"`
		AgeEnd       int                 `json:"ageEnd"`
		Distribution chanceDictDiagSlice `json:"diagnosedChance"`
		agentCounter int
	}

	// ConfigRandomFormat stores the entire file
	ConfigRandomFormat struct {
		IrregularLocChance      float64     `json:"irregulalLocationChance"`
		LocationTypeDistibution chanceSlice `json:"locationTypeDistibution"`
		PreCondDistibution      chanceSlice `json:"preCondDistibution"`
		StateDistribution       stateSlice  `json:"stateDistibution"`
		AgentTypeDistribution   chanceSlice `json:"agentTypeDistribution"`
		agentCounter            int
		locationCounter         int
		usedLocationCounter     int
		locationMap             map[string]string
		signalLocation          chan bool
	}

	// AgeInterval is used to define what are the intervals to get the state distributions
	AgeInterval struct {
		Begin int
		End   int
	}
)

func makeConfigRandomformat(ages []AgeInterval) ConfigRandomFormat {
	newCFG := ConfigRandomFormat{
		StateDistribution: make(stateSlice, len(ages)),
		locationMap:       make(map[string]string),
		signalLocation:    make(chan bool),
	}
	for idx := range newCFG.StateDistribution {
		newCFG.StateDistribution[idx].AgeStart = ages[idx].Begin
		newCFG.StateDistribution[idx].AgeEnd = ages[idx].End
	}
	return newCFG
}

func readJSON(filePath string) (map[string]interface{}, error) {
	result := make(map[string]interface{})
	file, err := os.Open(filePath)
	if err != nil {
		return result, err
	}
	defer file.Close()
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return result, err
	}
	err = json.Unmarshal(data, &result)
	return result, err
}

func mapGet(dict map[string]interface{}, key string) interface{} {
	ret, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	return ret
}

func mapGetString(dict map[string]interface{}, key string) string {
	tmp, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	switch v := tmp.(type) {
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', 0, 64)
	default:
		panic(fmt.Errorf("Key (%s) could not transform directly or indirectly to string", key))
	}
}

func mapGetInt(dict map[string]interface{}, key string) int {
	tmp, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	switch v := tmp.(type) {
	case float64:
		return int(v)
	case string:
		vInteger, err := strconv.Atoi(v)
		if err != nil {
			panic(fmt.Errorf("String to int conversion error for key (%s), error is : %v", key, err))
		}
		return vInteger
	default:
		panic(fmt.Errorf("Key (%s) could not transform directly or indirectly to int", key))
	}
}

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
			Value:      state,
			Chance:     1.0,
			DiagChance: 0.0,
		})
	}
}

func (sd *stateSlice) increment(age int, state string) {
	for i, value := range *sd {
		if value.AgeStart < age && age < value.AgeEnd {
			(*sd)[i].agentCounter++
			(*sd)[i].Distribution.increment(state)
		}
	}
}

func (crf *ConfigRandomFormat) addPerson(person map[string]interface{}) {
	crf.agentCounter++
	crf.PreCondDistibution.increment(mapGetString(person, "preCond"))
	crf.StateDistribution.increment(mapGetInt(person, "age"), mapGetString(person, "state"))
	crf.AgentTypeDistribution.increment(mapGetString(person, "typeID"))
}

func (crf *ConfigRandomFormat) addLocation(location map[string]interface{}) {
	crf.locationCounter++
	crf.LocationTypeDistibution.increment(mapGetString(location, "type"))
}

func (crf *ConfigRandomFormat) calculateIrregularLocations(agents []interface{}, wg *sync.WaitGroup) {
	defer wg.Done()
	ok := <-crf.signalLocation
	utils.InfoLogger.Println("Starting to calculate irregular locations")
	if !ok {
		return
	}
	for _, person := range agents {
		locations := mapGet(person.(map[string]interface{}), "locations").([]interface{})
		for _, loc := range locations {
			crf.usedLocationCounter++
			locMap := loc.(map[string]interface{})
			locID := mapGetString(locMap, "locID")
			originalTypeID, ok := crf.locationMap[locID]
			if !ok {
				panic(fmt.Errorf("Location ID (%s) in agents file does not exists in locations file", locID))
			}
			if originalTypeID != mapGetString(locMap, "typeID") {
				crf.IrregularLocChance++
			}
		}
	}
	utils.InfoLogger.Println("Finished calculating irregular locations")
}

func (crf *ConfigRandomFormat) readAgents(file string, wg *sync.WaitGroup) {
	defer wg.Done()
	utils.InfoLogger.Println("Parsing agents file")
	agentsData, err := readJSON(file)
	if err != nil {
		panic(err)
	}
	people := mapGet(agentsData, "people").([]interface{})
	wg.Add(1)
	go crf.calculateIrregularLocations(people, wg)
	for _, person := range people {
		crf.addPerson(person.(map[string]interface{}))
	}
	utils.InfoLogger.Println("Finished parsing agents file")
}

func (crf *ConfigRandomFormat) readLocations(file string, wg *sync.WaitGroup) {
	defer wg.Done()
	isChannelClosed := false
	isChannelClosedPtr := &isChannelClosed
	defer func(channelClosed *bool) {
		if err := recover(); err != nil {
			if !(*channelClosed) {
				crf.signalLocation <- false
				close(crf.signalLocation)
			}
			panic(err)
		}
	}(isChannelClosedPtr)
	utils.InfoLogger.Println("Parsing locations file")
	locationsData, err := readJSON(file)
	if err != nil {
		panic(err)
	}
	places := mapGet(locationsData, "places").([]interface{})
	for _, location := range places {
		tmp := location.(map[string]interface{})
		crf.locationMap[mapGetString(tmp, "ID")] = mapGetString(tmp, "type")
	}
	crf.signalLocation <- true
	close(crf.signalLocation)
	isChannelClosed = true
	for _, location := range places {
		crf.addLocation(location.(map[string]interface{}))
	}
	utils.InfoLogger.Println("Finished parsing locations file")
}

// CreateConfigRandomData create configRandom data from the agents and locations JSON files
func CreateConfigRandomData(agentsFile string, locationFile string, ages []AgeInterval) (ConfigRandomFormat, error) {
	defer func() {
		if err := recover(); err != nil {
			utils.ErrorLogger.Println(err)
		}
	}()
	result := makeConfigRandomformat(ages)
	var wg sync.WaitGroup

	wg.Add(2)
	go result.readAgents(agentsFile, &wg)
	go result.readLocations(locationFile, &wg)
	wg.Wait()

	return result, nil
}

func (pcd *chanceSlice) divideChances(number int) {
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
	crf.IrregularLocChance /= float64(crf.usedLocationCounter)
	crf.AgentTypeDistribution.divideChances(crf.agentCounter)
	crf.LocationTypeDistibution.divideChances(crf.locationCounter)
	crf.PreCondDistibution.divideChances(crf.agentCounter)
	crf.StateDistribution.calculatePercentages()
}

// WriteToFile writes out it's data in JSON format to the file path
func (crf *ConfigRandomFormat) WriteToFile(file string) error {
	utils.InfoLogger.Println("Writing out file")
	crf.calculatePercentages()
	data, err := json.MarshalIndent(crf, "", "    ")
	if err != nil {
		return err
	}
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	defer f.Close()
	n, err := f.Write(data)
	if n != len(data) {
		return fmt.Errorf("Could not write every byte of data, for some reason")
	}
	return nil
}
