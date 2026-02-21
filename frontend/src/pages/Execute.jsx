import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../lib/api'
import { connectStream } from '../lib/stream'
import GateVisualizer from '../components/GateVisualizer'
import SealCard from '../components/SealCard'
import ChainView from '../components/ChainView'

export default function Execute() {
  const [messages, setMessages] = useState([])
  const [currentGates, setCurrentGates] = useState({})
  const [currentSeals, setCurrentSeals] = useState([])
  const [currentStep, setCurrentStep] = useState(-1)
  const [streaming, setStreaming] = useState(false)
  const [input, setInput] = useState('')
  const [personas, setPersonas] = useState([])
  const [selectedPersona, setSelectedPersona] = useState('')
  const closeStreamRef = useRef(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    api.get('/personas')
      .then(data => {
        const list = Array.isArray(data) ? data : (data.personas || [])
        setPersonas(list)
        if (list.length > 0) {
          setSelectedPersona(list[0].name || list[0].persona_name || '')
        }
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    return () => {
      if (closeStreamRef.current) closeStreamRef.current()
    }
  }, [])

  function handleSubmit() {
    const task = input.trim()
    if (!task || streaming) return

    setMessages(prev => [...prev, { role: 'user', content: task }])
    setInput('')
    setCurrentGates({})
    setCurrentSeals([])
    setCurrentStep(0)
    setStreaming(true)

    const close = connectStream('/v1/execute/stream', {
      task,
      persona_name: selectedPersona,
    }, {
      onChainStarted: () => {
        setCurrentStep(0)
      },
      onStepStarted: (data) => {
        setCurrentStep(data.step_index ?? data.step ?? 0)
      },
      onGateResult: (data) => {
        setCurrentGates(prev => ({
          ...prev,
          [data.gate_name]: { verdict: data.verdict, score: data.score },
        }))
      },
      onSealCreated: (data) => {
        setCurrentSeals(prev => [...prev, data])
      },
      onStepCompleted: () => {},
      onChainCompleted: (data) => {
        const text = data.summary || data.result || 'Chain completed.'
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: text,
          gatesSummary: summarizeGates(currentGatesRef.current),
        }])
        setStreaming(false)
      },
      onError: (err) => {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${err.message || 'Stream failed'}`,
          isError: true,
        }])
        setStreaming(false)
      },
    })

    closeStreamRef.current = close
  }

  // Keep a ref to currentGates so the onChainCompleted handler can read the latest value
  const currentGatesRef = useRef(currentGates)
  useEffect(() => {
    currentGatesRef.current = currentGates
  }, [currentGates])

  function summarizeGates(gates) {
    const names = ['scope', 'intent', 'ttl', 'drift']
    let passed = 0
    let total = 0
    for (const name of names) {
      const g = gates[name]
      if (g && g.verdict !== undefined && g.verdict !== null) {
        total++
        if (g.verdict === 'pass' || g.verdict === true) passed++
      }
    }
    if (total === 0) return null
    return `${passed}/${total} gates passed`
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      <div className="p-6 border-b border-gray-800">
        <h1 className="text-2xl font-bold">Execute</h1>
      </div>

      <div className="flex gap-6 p-6 flex-1 min-h-0">
        {/* Left column: Chat */}
        <div className="flex flex-col flex-1 bg-gray-900 border border-gray-800 rounded-lg min-h-0">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && !streaming && (
              <div className="flex items-center justify-center h-full">
                <p className="text-gray-500 text-sm">Enter a task below to begin execution.</p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-indigo-600 text-white'
                      : msg.isError
                        ? 'bg-red-900/50 border border-red-800 text-red-200'
                        : 'bg-gray-800 text-gray-100'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  {msg.gatesSummary && (
                    <p className="text-xs text-gray-400 mt-2">{msg.gatesSummary}</p>
                  )}
                </div>
              </div>
            ))}

            {streaming && (
              <div className="flex justify-start">
                <div className="bg-gray-800 text-gray-400 rounded-lg px-4 py-3 text-sm animate-pulse">
                  Streaming...
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="border-t border-gray-800 p-4 flex gap-3">
            <select
              value={selectedPersona}
              onChange={e => setSelectedPersona(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-3 text-white text-sm"
            >
              {personas.map(p => {
                const name = p.name || p.persona_name
                return (
                  <option key={name} value={name}>{name}</option>
                )
              })}
              {personas.length === 0 && (
                <option value="">No personas</option>
              )}
            </select>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter a task..."
              disabled={streaming}
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500 transition-colors disabled:opacity-50"
            />
            <button
              onClick={handleSubmit}
              disabled={streaming || !input.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 px-6 py-3 rounded-lg font-medium transition-colors"
            >
              Send
            </button>
          </div>
        </div>

        {/* Right column: Live execution view */}
        <div className="w-96 space-y-4 flex-shrink-0">
          <GateVisualizer gates={currentGates} />

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Current Chain</h3>
            {currentSeals.length > 0 ? (
              <ChainView seals={currentSeals} currentStep={currentStep} />
            ) : (
              <p className="text-xs text-gray-500">No active chain. Submit a task to begin.</p>
            )}
          </div>

          {currentSeals.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-300">Seals</h3>
              {currentSeals.map((seal, i) => (
                <SealCard key={seal.seal_id || seal.id || i} seal={seal} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
