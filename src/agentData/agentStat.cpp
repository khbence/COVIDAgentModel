#include "agentStats.h"

std::ostream& operator<<(std::ostream& os, const AgentStats& s) {
    if (s.infectedTimestamp == std::numeric_limits<decltype(s.infectedTimestamp)>::max()) { return os; }
    os << "Infected at " << s.infectedTimestamp << " location " << s.infectedLocation << " diagnosed: " << s.diagnosedTimestamp
       << " quarantined: " << s.quarantinedTimestamp;
    os << " worst state " << static_cast<unsigned>(s.worstState) << " between: " << s.worstStateTimestamp << "-" << s.worstStateEndTimestamp << "\n";
    return os;
}