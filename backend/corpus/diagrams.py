"""
corpus/diagrams.py — the seed blueprint corpus.

Five hand-authored, structurally different text drawings.  They exist so the
system has something non-trivial to reason over the moment it boots, and so the
spatial chunker is exercised across genuinely different layout grammars:
box-and-arrow, bus topology, architectural plan, state machine, and elevation.
"""
from __future__ import annotations

from typing import Dict, List

PAYMENTS = r"""
                     GLYPHMIND REFERENCE TOPOLOGY  ::  payments-platform  rev.7

       +--------------+                             +------------------+
       |   Internet   |                             |   Status Page    |
       +------+-------+                             +--------+---------+
              |                                              ^
              | tls                                          | health
              v                                              |
       +------+-------+        http          +---------------+------+
       |   Edge CDN   +--------------------->+     API Gateway      |
       +------+-------+                      +--+----------------+--+
              |                                 |                |
              | miss                            | grpc           | grpc
              v                                 v                v
       +------+-------+                  +------+------+   +-----+-------+
       | Object Store |                  |  Auth Svc   |   | Ledger Svc  |
       +--------------+                  +------+------+   +-----+-------+
                                                |                |
                                                | sql            | sql
                                                v                v
                                         +------+------+   +-----+-------+
                                         |   User DB   |   |  Ledger DB  |
                                         +------+------+   +-----+-------+
                                                |                |
                                                +-------+--------+
                                                        | wal ship
                                                        v
                                                +-------+--------+
                                                | Audit Archive  |
                                                +----------------+

       legend:  --->  synchronous call        every edge is TLS terminated
"""

CAMPUS = r"""
                      CAMPUS NETWORK  --  building C  --  vlan plan

                              +-------------------+
                              |   Border Router   |
                              +---------+---------+
                                        |
                                 bgp    | uplink
                                        v
                              +---------+---------+
                              |   Core Switch     |
                              +--+-----+-------+--+
                                 |     |       |
                  trunk 10g      |     | trunk |      trunk 10g
         +-----------------------+     |       +------------------------+
         |                             |                                |
         v                             v                                v
+--------+--------+          +---------+--------+            +----------+-------+
| Access Switch A |          | Access Switch B  |            | Access Switch C  |
+--------+--------+          +---------+--------+            +----------+-------+
         |                             |                                |
         | vlan 10                     | vlan 20                        | vlan 30
         v                             v                                v
+--------+--------+          +---------+--------+            +----------+-------+
|   Staff  WiFi   |          |   Lab Subnet     |            |  Guest Subnet    |
+-----------------+          +---------+--------+            +----------+-------+
                                       |                                |
                                       | acl                            | acl
                                       v                                v
                              +--------+---------+            +---------+--------+
                              |    Firewall      +<-----------+   Captive Portal |
                              +------------------+   policy   +------------------+

    note: guest traffic is forced through the captive portal before the firewall
"""

FLOORPLAN = r"""
      LEVEL 2  ::  NORTH WING  ::  scale 1 square = 1 metre

      +---------------------------------------------------------------+
      |                                                               |
      |  +-------------+   +-------------+        +----------------+  |
      |  |  Lab 201    |   |  Lab 203    |        |  Server Room   |  |
      |  |             |   |             |        |                |  |
      |  +------+------+   +------+------+        +--------+-------+  |
      |         |                 |                        |         |
      |         d1                d2                       d3        |
      |         |                 |                        |         |
      |  =======+=================+========================+=======  |
      |                      CORRIDOR                                 |
      |  =======+=================+========================+=======  |
      |         |                 |                        |         |
      |         d4                d5                       d6        |
      |         |                 |                        |         |
      |  +------+------+   +------+------+        +--------+-------+  |
      |  |  Office 202 |   |  Office 204 |        |   Break Room   |  |
      |  |             |   |             |        |                |  |
      |  +-------------+   +-------------+        +----------------+  |
      |                                                               |
      |                                        +-------------------+  |
      |                                        |   Fire Stair  S2  |  |
      |                                        +-------------------+  |
      +---------------------------------------------------------------+

      d1..d6 are doorways.  The server room is the only space on this level
      with a dedicated HVAC return.
"""

STATE_MACHINE = r"""
                    ORDER LIFECYCLE  --  finite state machine

        [ DRAFT ] ---- submit ----> [ PENDING ] ---- authorise ----> [ PAID ]
            ^                            |                              |
            |                            |                              |
            |                         decline                        fulfil
            |                            |                              |
            |                            v                              v
            +-------- amend -------- [ FAILED ]                   [ SHIPPED ]
                                         |                              |
                                      retry                          deliver
                                         |                              |
                                         v                              v
                                    [ PENDING ]                  [ COMPLETED ]
                                                                        |
                                                                      refund
                                                                        |
                                                                        v
                                                                  [ REFUNDED ]

        terminal states:  COMPLETED, REFUNDED
        DRAFT is the only state reachable by amendment.
"""

RACK = r"""
            DC-WEST  ::  RACK A14  ::  front elevation  ::  42U

            +--------------------------------------------------+
       42U  |            Blanking Panel                        |
            +--------------------------------------------------+
       41U  |  +--------------------------------------------+  |
            |  |        ToR Switch  sw-a14-01   48x25G      |  |
       40U  |  +--------------------------------------------+  |
            +--------------------------------------------------+
       38U  |  +--------------------------------------------+  |
            |  |        ToR Switch  sw-a14-02   48x25G      |  |
       37U  |  +--------------------------------------------+  |
            +--------------------------------------------------+
       30U  |  +--------------------------------------------+  |
            |  |        Compute  cn-a14-001 .. cn-a14-012   |  |
       18U  |  +--------------------------------------------+  |
            +--------------------------------------------------+
       16U  |  +--------------------------------------------+  |
            |  |        Storage  st-a14-01   90 bay JBOD    |  |
       10U  |  +--------------------------------------------+  |
            +--------------------------------------------------+
        4U  |  +--------------------------------------------+  |
            |  |        PDU A  /  PDU B    dual feed        |  |
        1U  |  +--------------------------------------------+  |
            +--------------------------------------------------+

            thermal note: compute block draws 11.4 kW at p95
"""


SEED_DIAGRAMS: List[Dict[str, str]] = [
    {"name": "Payments Platform Topology", "kind": "topology", "text": PAYMENTS,
     "blurb": "Box-and-arrow microservice architecture with directed, labelled edges.",
     "samples": [
         "What connects to the API Gateway?",
         "Is there a path from the Internet to the Audit Archive?",
         "What sits below the Ledger Svc?",
         "Which component is the biggest single point of failure?",
     ]},
    {"name": "Campus Network — Building C", "kind": "topology", "text": CAMPUS,
     "blurb": "Bus-style network plan with trunks, VLANs and a policy loop.",
     "samples": [
         "How many access switches are there?",
         "What is the path from the Guest Subnet to the Firewall?",
         "What is to the right of Access Switch A?",
         "List every element in the diagram.",
     ]},
    {"name": "Level 2 Floor Plan — North Wing", "kind": "floorplan", "text": FLOORPLAN,
     "blurb": "Architectural plan: rooms, a corridor bus, and six doorways.",
     "samples": [
         "What rooms are above the corridor?",
         "What is inside the north wing?",
         "What is to the right of Lab 203?",
         "How many rooms are on this level?",
     ]},
    {"name": "Order Lifecycle State Machine", "kind": "state", "text": STATE_MACHINE,
     "blurb": "Bracket-notation FSM with labelled transitions and terminal states.",
     "samples": [
         "What are the transitions out of PENDING?",
         "Is there a cycle in this state machine?",
         "Path from DRAFT to REFUNDED?",
         "Which states are terminal?",
     ]},
    {"name": "Rack A14 Elevation", "kind": "rack", "text": RACK,
     "blurb": "Nested-rectangle elevation drawing — tests containment reasoning.",
     "samples": [
         "What is inside rack A14?",
         "What is below the compute block?",
         "How many devices are mounted?",
         "What is at the top of the rack?",
     ]},
]
