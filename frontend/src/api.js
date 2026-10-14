import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const browse = (path) => api.get('/browse', { params: path ? { path } : {} }).then(r => r.data)
export const wavCount = (path, extra = false) =>
  api.get('/wav-count', { params: { path, extra } }).then(r => r.data)
export const getPurposes = () => api.get('/purposes').then(r => r.data)
export const openFolder = (path) => api.post('/open-folder', { path }).then(r => r.data)

export const startRun = (payload) => api.post('/run', payload).then(r => r.data)
export const getJob = (jobId) => api.get(`/jobs/${jobId}`).then(r => r.data)
export const getJobResults = (jobId) => api.get(`/jobs/${jobId}/results`).then(r => r.data)
export const downloadUrl = (jobId) => `/api/jobs/${jobId}/download`

export const errorMessage = (e, fallback = 'Something went wrong.') =>
  e?.response?.data?.detail || e?.message || fallback

export default api
