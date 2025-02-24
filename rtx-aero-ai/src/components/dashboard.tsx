"use client"

import React, { useState, useEffect, useMemo } from "react"
import { MapContainer, TileLayer, Marker, Popup, Circle, ScaleControl, useMap, useMapEvents } from "react-leaflet"
import L from "leaflet"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { History, ThumbsUp, ThumbsDown, Plane, AlertTriangle, Info, Plus, Minus } from "lucide-react"

// Leaflet CSS
import "leaflet/dist/leaflet.css"

// Marine Corps Air Station El Toro coordinates (near Irvine, CA)
const BASE_COORDINATES: [number, number] = [33.6761, -117.7317]

// Radius in miles
const RADIUS_MILES = 10
const RADIUS_KM = RADIUS_MILES * 1.60934

// Function to generate random coordinates within a radius
const generateRandomCoordinate = (center: [number, number], radiusInKm: number): [number, number] => {
  const radiusInDegrees = radiusInKm / 111.32 // 1 degree is approximately 111.32 km
  const u = Math.random()
  const v = Math.random()
  const w = radiusInDegrees * Math.sqrt(u)
  const t = 2 * Math.PI * v
  const x = w * Math.cos(t)
  const y = w * Math.sin(t)
  return [center[0] + y, center[1] + x]
}

// Function to generate a random update type
const generateUpdateType = () => {
  const types = ["A", "B", "C", "D"]
  return types[Math.floor(Math.random() * types.length)]
}

// Generate 20 random flights within the 10-mile radius
const generateFlights = (count: number) => {
  const types = ["Commercial", "Private", "Military", "UAV"]
  const navModes = ["autopilot", "lnav", "tcas", "vnav"]

  return Array.from({ length: count }, (_, i) => {
    const threatScore = Math.random()
    const [lat, lon] = generateRandomCoordinate(BASE_COORDINATES, RADIUS_KM)

    return {
      id: `AC${String(i + 1).padStart(3, "0")}`,
      pitr: new Date().toISOString(),
      type: types[Math.floor(Math.random() * types.length)],
      ident: `FL${Math.floor(Math.random() * 1000)}`,
      air_ground: Math.random() > 0.1 ? "air" : "ground",
      alt: Math.round(Math.random() * 40000 + 5000),
      gs: Math.round(Math.random() * 400 + 100),
      heading: Math.round(Math.random() * 360),
      lat,
      lon,
      nav_modes: navModes[Math.floor(Math.random() * navModes.length)],
      squawk: Math.floor(Math.random() * 7777)
        .toString()
        .padStart(4, "0"),
      updateType: generateUpdateType(),
      vertRate: Math.round(Math.random() * 2000 - 1000),
      threat: threatScore > 0.7 ? "High" : threatScore > 0.3 ? "Medium" : "Low",
      threatScore,
      previousData: null,
    }
  })
}

const initialAircraft = generateFlights(20)

const restrictedZone = {
  position: BASE_COORDINATES,
  radius: 2000, // meters
}

function MapControls({ selectedAircraft }: { selectedAircraft: any }) {
  const map = useMap()

  useEffect(() => {
    if (selectedAircraft) {
      map.flyTo([selectedAircraft.lat, selectedAircraft.lon], map.getZoom())
    } else {
      map.flyTo(BASE_COORDINATES, 12)
    }
  }, [selectedAircraft, map])

  return null
}

function ZoomControl() {
  const map = useMapEvents({
    zoomend: () => {
      console.log(map.getZoom())
    },
  })

  return (
    <div className="leaflet-top leaflet-left">
      <div className="leaflet-control leaflet-bar">
        <Button className="p-0 w-8 h-8" variant="outline" onClick={() => map.zoomIn()}>
          <Plus className="h-4 w-4" />
        </Button>
        <Button className="p-0 w-8 h-8 border-t-0" variant="outline" onClick={() => map.zoomOut()}>
          <Minus className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

interface Aircraft {
  id: string;
  pitr: string;
  type: string;
  ident: string;
  air_ground: string;
  alt: number;
  gs: number;
  heading: number;
  lat: number;
  lon: number;
  nav_modes: string;
  squawk: string;
  updateType: string;
  vertRate: number;
  threat: string;
  threatScore: number;
  previousData: Aircraft | null;
}

export default function Dashboard() {
  const [aircraft, setAircraft] = useState<Aircraft[]>(initialAircraft)
  const [selectedAircraft, setSelectedAircraft] = useState<Aircraft | null>(null)
  const [trainingMode, setTrainingMode] = useState(false)
  const [showRestrictedZone, setShowRestrictedZone] = useState(true)
  const [threatThreshold, setThreatThreshold] = useState(0.5)
  const [mapCenter] = useState(BASE_COORDINATES)
  const [mapZoom] = useState(12)

  useEffect(() => {
    const interval = setInterval(() => {
      setAircraft((prevAircraft) =>
        prevAircraft.map((ac) => {
          const newData = {
            ...ac,
            pitr: new Date().toISOString(),
            alt: ac.alt + Math.round(Math.random() * 200 - 100),
            gs: ac.gs + Math.round(Math.random() * 20 - 10),
            heading: (ac.heading + Math.round(Math.random() * 10 - 5) + 360) % 360,
            lat: ac.lat + (Math.random() - 0.5) * 0.01,
            lon: ac.lon + (Math.random() - 0.5) * 0.01,
            vertRate: Math.round(Math.random() * 2000 - 1000),
            updateType: generateUpdateType(),
            previousData: { ...ac },
          }
          return newData
        }),
      )
    }, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [])

  const getAircraftIcon = (aircraft: Aircraft) => {
    const color = aircraft.threat === "High" ? "red" : aircraft.threat === "Medium" ? "orange" : "green"
    return L.divIcon({
      className: "custom-icon",
      html: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}" width="24" height="24">
               <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/>
             </svg>`,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    })
  }

  const handleAircraftClick = (aircraft: Aircraft) => {
    setSelectedAircraft(aircraft)
  }

  const handleFeedback = (feedback: "correct" | "incorrect") => {
    console.log(`Feedback for ${selectedAircraft?.id}: ${feedback}`)
    // Here you would typically send this feedback to your backend
  }

  const getUpdateTypeDescription = (updateType: string) => {
    switch (updateType) {
      case "A":
        return "Position Update"
      case "B":
        return "Velocity Update"
      case "C":
        return "Identification Update"
      case "D":
        return "Other Data Update"
      default:
        return "Unknown Update Type"
    }
  }

  const getThreatExplanation = (aircraft: Aircraft) => {
    if (aircraft.threat === "High") {
      return `Aircraft ${aircraft.id} is flagged as a high threat due to unusual behavior patterns. It's currently at altitude ${aircraft.alt} ft with a vertical rate of ${aircraft.vertRate} ft/min. Confidence: High`
    } else if (aircraft.threat === "Medium") {
      return `Aircraft ${aircraft.id} is showing slight deviations from expected patterns. It's traveling at ${aircraft.gs} knots. Confidence: Medium`
    }
    return `Aircraft ${aircraft.id} is behaving within normal parameters. Current altitude: ${aircraft.alt} ft, speed: ${aircraft.gs} knots. Confidence: Low`
  }

  const getChanges = (aircraft: Aircraft) => {
    if (!aircraft.previousData) return []
    const changes = []
    if (Math.abs(aircraft.gs - aircraft.previousData.gs) > 20) {
      changes.push(`Ground speed changed by ${(aircraft.gs - aircraft.previousData.gs).toFixed(2)} knots`)
    }
    if (Math.abs(aircraft.alt - aircraft.previousData.alt) > 500) {
      changes.push(`Altitude changed by ${(aircraft.alt - aircraft.previousData.alt).toFixed(2)} feet`)
    }
    if (Math.abs(aircraft.heading - aircraft.previousData.heading) > 15) {
      changes.push(
        `Heading changed by ${Math.abs(aircraft.heading - aircraft.previousData.heading).toFixed(2)} degrees`,
      )
    }
    return changes
  }

  const sortedAircraft = useMemo(() => {
    return [...aircraft].sort((a, b) => b.threatScore - a.threatScore)
  }, [aircraft])

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm z-10 py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-gray-900">ADGE Threat Detection Dashboard</h1>
          <div className="flex items-center space-x-4">
            <Switch checked={trainingMode} onCheckedChange={setTrainingMode} id="training-mode" />
            <label htmlFor="training-mode">Training Mode</label>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden bg-gray-200 p-4">
        <div className="h-full flex gap-4">
          {/* Map View (Larger portion) */}
          <div className="flex-grow">
            <Card className="h-full">
              <CardContent className="p-0 h-full">
                <MapContainer
                  center={mapCenter}
                  zoom={mapZoom}
                  style={{ height: "100%", width: "100%" }}
                  zoomControl={false}
                  minZoom={11}
                  maxZoom={14}
                >
                  <TileLayer
                    url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
                  />
                  <ScaleControl position="bottomleft" imperial={true} />
                  <ZoomControl />
                  {aircraft.map((ac) => (
                    <React.Fragment key={ac.id}>
                      <Marker
                        position={[ac.lat, ac.lon]}
                        icon={getAircraftIcon(ac)}
                        eventHandlers={{
                          click: () => handleAircraftClick(ac),
                        }}
                      >
                        <Popup>
                          <div className="text-sm">
                            <p className="font-bold">
                              {ac.id} - {ac.type}
                            </p>
                            <p>Threat: {ac.threat}</p>
                            <p>Altitude: {ac.alt} ft</p>
                            <p>Speed: {ac.gs} knots</p>
                          </div>
                        </Popup>
                      </Marker>
                      {selectedAircraft && selectedAircraft.id === ac.id && (
                        <Circle
                          center={[ac.lat, ac.lon]}
                          radius={1000}
                          pathOptions={{ color: "blue", fillColor: "blue", fillOpacity: 0.2 }}
                        />
                      )}
                    </React.Fragment>
                  ))}
                  {showRestrictedZone && (
                    <Circle
                      center={restrictedZone.position}
                      radius={restrictedZone.radius}
                      pathOptions={{ color: "yellow", fillColor: "yellow", fillOpacity: 0.2 }}
                    />
                  )}
                  <MapControls selectedAircraft={selectedAircraft} />
                </MapContainer>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="w-96 space-y-4">
            {/* Threat Detection Controls */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Threat Detection Settings</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Threat Threshold</label>
                    <Slider
                      value={[threatThreshold]}
                      onValueChange={(value) => setThreatThreshold(value[0])}
                      max={1}
                      step={0.01}
                    />
                    <span className="text-sm text-gray-500">{threatThreshold.toFixed(2)}</span>
                  </div>
                  <Button className="w-full">Apply Settings</Button>
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => setShowRestrictedZone(!showRestrictedZone)}
                  >
                    {showRestrictedZone ? "Hide" : "Show"} Restricted Zone
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Aircraft List */}
            <Card className="flex-grow">
              <CardHeader>
                <CardTitle className="text-lg">Aircraft List</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[calc(100vh-400px)]">
                  {sortedAircraft.map((ac) => (
                    <div
                      key={ac.id}
                      className={`p-4 border-b cursor-pointer hover:bg-gray-100 ${selectedAircraft?.id === ac.id ? "bg-blue-50" : ""}`}
                      onClick={() => handleAircraftClick(ac)}
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-semibold">{ac.id}</span>
                        <Badge
                          variant={
                            ac.threat === "High" ? "destructive" : ac.threat === "Medium" ? "secondary" : "outline"
                          }
                        >
                          {ac.threat}
                        </Badge>
                      </div>
                      <div className="text-sm text-gray-600">
                        <p>
                          {ac.type} | Alt: {ac.alt} ft | Speed: {ac.gs} knots
                        </p>
                        <p>
                          Heading: {ac.heading}° | Vert. Rate: {ac.vertRate} ft/min
                        </p>
                      </div>
                    </div>
                  ))}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Footer - Selected Aircraft Details */}
      <footer className="bg-white border-t shadow-lg p-4">
        {selectedAircraft ? (
          <>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Selected Aircraft: {selectedAircraft.id}</h2>
              <Button variant="outline" size="sm">
                <History className="mr-2 h-4 w-4" /> View History
              </Button>
            </div>
            <Tabs defaultValue="details">
              <TabsList>
                <TabsTrigger value="details">Details</TabsTrigger>
                <TabsTrigger value="analysis">AI Analysis</TabsTrigger>
                <TabsTrigger value="actions">Actions</TabsTrigger>
              </TabsList>
              <TabsContent value="details">
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p>
                      <strong>Type:</strong> {selectedAircraft.type}
                    </p>
                    <p>
                      <strong>Ident:</strong> {selectedAircraft.ident}
                    </p>
                    <p>
                      <strong>Ground Speed:</strong> {selectedAircraft.gs} knots
                    </p>
                    <p>
                      <strong>Altitude:</strong> {selectedAircraft.alt} ft
                    </p>
                    <p>
                      <strong>Vertical Rate:</strong> {selectedAircraft.vertRate} ft/min
                    </p>
                  </div>
                  <div>
                    <p>
                      <strong>Heading:</strong> {selectedAircraft.heading}°
                    </p>
                    <p>
                      <strong>Lat:</strong> {selectedAircraft.lat.toFixed(4)}
                    </p>
                    <p>
                      <strong>Lon:</strong> {selectedAircraft.lon.toFixed(4)}
                    </p>
                    <p>
                      <strong>Nav Mode:</strong> {selectedAircraft.nav_modes}
                    </p>
                    <p>
                      <strong>Squawk:</strong> {selectedAircraft.squawk}
                    </p>
                  </div>
                  <div>
                    <p>
                      <strong>Air/Ground:</strong> {selectedAircraft.air_ground}
                    </p>
                    <p>
                      <strong>Update Type:</strong> {getUpdateTypeDescription(selectedAircraft.updateType)}
                    </p>
                    <p>
                      <strong>Last Update:</strong> {new Date(selectedAircraft.pitr).toLocaleTimeString()}
                    </p>
                    <p>
                      <strong>Threat Level:</strong> {selectedAircraft.threat}
                    </p>
                    <p>
                      <strong>Threat Score:</strong> {selectedAircraft.threatScore.toFixed(2)}
                    </p>
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="analysis">
                <div className="space-y-4">
                  <p className="text-sm text-gray-600">{getThreatExplanation(selectedAircraft)}</p>
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="text-yellow-500" />
                    <span className="font-semibold">Recent Changes:</span>
                  </div>
                  <ul className="list-disc pl-5">
                    {getChanges(selectedAircraft).map((change, index) => (
                      <li key={index}>{change}</li>
                    ))}
                  </ul>
                  <div className="flex items-center space-x-2">
                    <Info className="text-blue-500" />
                    <span className="font-semibold">Recommendation:</span>
                    <span>
                      {selectedAircraft.threat === "High"
                        ? "Immediate attention required. Verify aircraft identity and intentions."
                        : selectedAircraft.threat === "Medium"
                          ? "Monitor closely for the next 15 minutes"
                          : "Continue routine monitoring"}
                    </span>
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="actions">
                <div className="space-y-4">
                  <Button className="w-full">
                    <Plane className="mr-2 h-4 w-4" /> Request Identification
                  </Button>
                  <Button className="w-full" variant="destructive">
                    <AlertTriangle className="mr-2 h-4 w-4" /> Escalate to Command
                  </Button>
                  <div className="flex justify-between">
                    <Button variant="outline" onClick={() => handleFeedback("correct")}>
                      <ThumbsUp className="mr-2 h-4 w-4" /> Correct Assessment
                    </Button>
                    <Button variant="outline" onClick={() => handleFeedback("incorrect")}>
                      <ThumbsDown className="mr-2 h-4 w-4" /> Incorrect Assessment
                    </Button>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </>
        ) : (
          <p className="text-center text-gray-500">Select an aircraft to view details</p>
        )}
      </footer>
    </div>
  )
}

