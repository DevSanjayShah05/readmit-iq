import { PatientForm } from "./components/PatientForm";


function App() {
  return (
    <div className="min-h-screen bg-slate-50 py-12">
      <div className="max-w-2xl mx-auto px-4">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold text-slate-900">ReadmitIQ</h1>
          <p className="text-slate-600 mt-1">
            30-day hospital readmission risk prediction with explainable ML.
          </p>
        </header>

        <PatientForm />
      </div>
    </div>
  );
}

export default App;
