import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function ProjectDescription() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Airborne Threat Detection System</h1>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="architecture">Architecture</TabsTrigger>
          <TabsTrigger value="requirements">Requirements</TabsTrigger>
          <TabsTrigger value="development">Development</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>Project Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <h2 className="text-xl font-semibold mb-2">Motivation/Background</h2>
              <p className="mb-4">
                This project addresses the challenge of detecting airborne threats within high-traffic airspaces. It
                proposes implementing an AI/ML model trained on historical data to assist operators in identifying
                airborne objects that demonstrate atypical behavior.
              </p>

              <h2 className="text-xl font-semibold mb-2">State of the Art / Innovation</h2>
              <p className="mb-4">
                Current threat detection relies on radar systems, human monitoring, and rule-based algorithms. This
                project innovates by using AI and machine learning to analyze historical data and detect atypical flight
                patterns in real-time.
              </p>

              <h2 className="text-xl font-semibold mb-2">Project Goals</h2>
              <ul className="list-disc pl-5 mb-4">
                <li>Gain hands-on experience using machine learning technologies</li>
                <li>Create an MVP of a machine learning model for detecting flight anomalies</li>
                <li>Build a React-based dashboard for real-time threat visualization</li>
                <li>Implement intuitive threat scoring display and alert system</li>
                <li>Design an offline-capable containerized application architecture</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="architecture">
          <Card>
            <CardHeader>
              <CardTitle>System Architecture</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4">
                The system processes FlightAware data, which comes from aircraft transponder signals picked up by
                secondary surveillance radar (SSR). This information is sent to Raytheon Technologies Air Defense Ground
                Environment (ADGE) Systems for tracking and analysis.
              </p>
              <h2 className="text-xl font-semibold mb-2">User Interaction and Design</h2>
              <p>
                The UI displays aircraft tracks, highlights potential threats, and provides a threat evaluation table.
                For testing purposes, a simple graph showing incoming air traffic data with color-coded threat levels is
                planned.
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="requirements">
          <Card>
            <CardHeader>
              <CardTitle>Requirements</CardTitle>
            </CardHeader>
            <CardContent>
              <h2 className="text-xl font-semibold mb-2">User Stories</h2>
              <ul className="list-disc pl-5 mb-4">
                <li>Live, interactive map displaying aircraft positions and details</li>
                <li>Real-time processing of air traffic data</li>
                <li>Automatic flagging of unusual aircraft behavior</li>
                <li>Training mode with simulated threat scenarios</li>
                <li>Explanation of why a track is flagged as a threat</li>
                <li>Ability to provide feedback on flagged threats</li>
              </ul>

              <h2 className="text-xl font-semibold mb-2">Use Cases</h2>
              <ul className="list-disc pl-5">
                <li>Real-Time Anomaly Detection in Flight Paths</li>
                <li>Identification of Unauthorized Airspace Entry</li>
                <li>Dynamic Threat Assessment During High Traffic</li>
                <li>Historical Pattern Analysis for Predictive Threat Detection</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="development">
          <Card>
            <CardHeader>
              <CardTitle>Development and Testing</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4">
                The project codebase is hosted on GitHub:{" "}
                <a href="https://github.com/xKimChip/AeroAI" className="text-blue-500 hover:underline">
                  https://github.com/xKimChip/AeroAI
                </a>
              </p>
              <p className="mb-4">
                Testing includes unit testing (PyTest for AI model, Jest for UI), performance metrics tracking, and
                system integration tests. The project uses Python for AI/ML, JavaScript with React for UI, and Docker
                for containerized deployment.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

