import axios from 'axios'
const http = axios.create({ baseURL: '/api', timeout: 60000 })
export function search(query, topK = 8, withAnswer = true) {
  return http.post('/search', { query, top_k: topK, with_answer: withAnswer })
}
