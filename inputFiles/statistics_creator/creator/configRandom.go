package creator

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"statistics_creator/utils"
	"sync"
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

func (ig *irregularGlobal) addCase(expected, real string) {
	ig.usedLocationCounter++
	if !ig.Details.increment(expected, real) {
		return
	}
	ig.GeneralChance++
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
	if !ok {
		return
	}
	utils.InfoLogger.Println("Starting to calculate irregular locations")
	for _, person := range agents {
		locations := mapGet(person.(map[string]interface{}), "locations").([]interface{})
		for _, loc := range locations {
			locMap := loc.(map[string]interface{})
			locID := mapGetString(locMap, "locID")
			originalTypeID, ok := crf.locationMap[locID]
			if !ok {
				panic(fmt.Errorf("Location ID (%s) in agents file does not exists in locations file", locID))
			}
			crf.IrregularLocChance.addCase(mapGetString(locMap, "typeID"), originalTypeID)
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

// WriteToFile writes out it's data in JSON format to the file path
func (crf *ConfigRandomFormat) WriteToFile(file string) error {
	utils.InfoLogger.Println("Writing out file")
	utils.InfoLogger.Printf("Number of agents: %d", crf.agentCounter)
	utils.InfoLogger.Printf("Number of locations: %d", crf.locationCounter)
	utils.InfoLogger.Printf("Ratio agent/loc: %f", float64(crf.agentCounter)/float64(crf.locationCounter))
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
