/*
 * @Author       : LIN Guocheng
 * @Date         : 2024-09-20 04:05:27
 * @LastEditors  : LIN Guocheng
 * @LastEditTime : 2024-10-09 01:15:57
 * @FilePath     : /root/PPFP/modules/ipv4/RlPpfpRoutingTable.cc
 * @Description  : Probabilistic routing table in RouterRL.
 */
#include "RlPpfpRoutingTable.h"

RlPpfpRoutingTable *RlPpfpRoutingTable::ppfpRoutingTable = NULL;

RlPpfpRoutingTable::RlPpfpRoutingTable()
{
}

RlPpfpRoutingTable::~RlPpfpRoutingTable()
{
    if (zmq_socket) {
        zmq_socket->close();
        delete zmq_socket;
        zmq_socket = nullptr;
        zmq_context->close();
        delete zmq_context;
    }
    if (ppfpRoutingTable)
        delete ppfpRoutingTable;
}

/**
 * @brief Used to get the unique static instance, throws an exception if the instance is not initialized
 *
 * @return RlPpfpRoutingTable* Probability routing table
 */
RlPpfpRoutingTable *RlPpfpRoutingTable::getInstance()
{
    return ppfpRoutingTable;
}

/**
 * @brief Initialize the probabilistic routing table
 *
 * @param num               Number of nodes in the network topology
 * @param file              Initial routing probability
 * @param port              ZMQ communication port
 * @param overTime_v        Timeout value
 * @param totalStep_v       Total number of simulation steps
 * @param simMode_v         Simulation mode
 * @return RlPpfpRoutingTable* Initialized probabilistic routing table
 */
RlPpfpRoutingTable *RlPpfpRoutingTable::initTable(int num, const char *initRoutingTable_v, int port,
                                                  double overTime_v, int totalStep_v, int simMode_v)
{
    if (!ppfpRoutingTable) {
        ppfpRoutingTable = new RlPpfpRoutingTable();
        ppfpRoutingTable->setVals(port, num, initRoutingTable_v, overTime_v, totalStep_v,
                                  simMode_v);
        ppfpRoutingTable->initiate();
    }
    return ppfpRoutingTable;
}

/**
 * @brief Initializes the instance, establishes ZMQ communication with the Python side based on TCP, and allocates memory for statistics variables
 *
 */
void RlPpfpRoutingTable::initiate()
{

    RlProbabilisticRoutingTable::initiate();

    topoDstTable = (int **)malloc(nodeNum * sizeof(int *));
    memset(topoDstTable, 0, nodeNum);
    for (int i = 0; i < nodeNum; i++) {
        topoDstTable[i] = (int *)malloc(nodeNum * nodeNum * sizeof(int));
        for (int j = 0; j < nodeNum; j++)
            topoDstTable[i][j] = (i == j) ? 0 : INT_MAX;
    }
    // Initialize the global topology information matrix
    for (int src = 0; src < nodeNum; src++) {
        vector<bool> visited(nodeNum, false);
        queue<pair<int, int>> q;
        visited[src] = true;
        q.push({src, 0});

        while (!q.empty()) {
            pair<int, int> front = q.front();
            q.pop();
            int node = front.first;
            int dist = front.second;

            // Iterate over all neighboring nodes
            for (int next = 0; next < nodeNum; ++next) {
                if (!visited[next] && topo[node][next] >= 0) {
                    visited[next] = true;
                    q.push({next, dist + 1});
                    topoDstTable[src][next] = dist + 1;
                }
            }
        }
    }
}

/**
 * @brief Get the next hop based on the probability forwarding matrix
 *
 * @param nodeId    ID of the current node
 * @param dstNode   ID of the destination node
 * @return int      Next hop node
 */
int RlPpfpRoutingTable::getNextNode(int nodeId, int srcNode, int dstNode)
{
    if (allProb[nodeId][dstNode]) {
        return dstNode;
    } else {
        vector<int> candidateNodes;
        vector<float> candidateProbs;
        float probSum = 0.0;

        // The core of PPFP
        for (int i = 0; i < nodeNum; i++) {
            bool addCandidateNode = true;
            bool routingCondition1 = topoDstTable[i][dstNode] < topoDstTable[nodeId][dstNode];
            bool routingCondition2 = topoDstTable[i][dstNode] == topoDstTable[nodeId][dstNode]
                                     && topoDstTable[i][srcNode] < topoDstTable[nodeId][srcNode];
            addCandidateNode = routingCondition1 || routingCondition2;
            if (addCandidateNode) {
                candidateNodes.push_back(i);
                candidateProbs.push_back(allProb[nodeId][i]);
                probSum += allProb[nodeId][i];
            }
        }

        // Roulette wheel selection algorithm for probabilistic routing
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, probSum);
        float randProb = dis(gen);
        float curProb = 0.0;
        for (int i = 0; i < candidateNodes.size(); i++) {
            curProb += candidateProbs[i];
            if (randProb <= curProb)
                return candidateNodes[i];
        }
        return candidateNodes[0];
    }
}

/**
 * @brief Called after all packets of the current step have been sent, communicates with the ZMQ server (Python side), transfers the current step's throughput, obtains the weights of nodes for the next step, calculates the corresponding forwarding probabilities, and clears the already collected throughput data
 *
 * @param step  Current step number
 */
void RlPpfpRoutingTable::updateRoutingTable(int step, double stepTime)
{
    updateNodeCount[step]++;
    if (updateNodeCount[step] == nodeNum) {
        // The last node to enter the next step update
        string stateStr;
        stateStr += "s@@" + to_string(step) + "@@";
        for (int i = 0; i < nodeNum; i++)
            for (int j = 0; j < nodeNum; j++) {
                if (i == nodeNum - 1 && j == nodeNum - 1)
                    stateStr += to_string(double(pkct[i][j]) / 1000 / 1000 / stepTime);
                else
                    stateStr += to_string(double(pkct[i][j]) / 1000 / 1000 / stepTime) + ",";
            }
        clearPkts();
        cout << stateStr << endl;
        const char *reqData = stateStr.c_str();
        const size_t reqLen = strlen(reqData);
        zmq::message_t request{reqLen};
        memcpy(request.data(), reqData, reqLen);
        zmq_socket->send(request, zmq::send_flags::none);

        zmq::message_t reply;
        auto res = zmq_socket->recv(reply, zmq::recv_flags::none);
        char *buffer = new char[reply.size() + 1];
        memset(buffer, 0, reply.size() + 1);
        memcpy(buffer, reply.data(), reply.size());
        // The received data contains (edgeNum * 2) weights, assembled into a probability table in the inet side
        char *od_prob;
        double *weights = new double[edgeNum * 2];
        for (int i = 0; i < edgeNum * 2; i++) {
            if (i == 0) {
                od_prob = strtok(buffer, ",");
            } else {
                od_prob = strtok(NULL, ",");
            }
            weights[i] = atof(od_prob);
        }

        for (int row = 0; row < nodeNum; row++) {
            double totalWeight = 0.0;
            for (int col = 0; col < nodeNum; col++) {
                if (topo[row][col] != -1) {
                    totalWeight += weights[topo[row][col]];
                }
            }
            for (int col = 0; col < nodeNum; col++) {
                if (topo[row][col] != -1) {
                    float prob = (float)(weights[topo[row][col]] / totalWeight * 100);
                    allProb[row][col] = prob;
                }
            }
        }
        delete weights;
        sendId = 0; // Reset packet ID for the next step
        delete buffer;
    }
}