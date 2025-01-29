import { getFirestore, doc, getDoc, setDoc, updateDoc, increment, serverTimestamp } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-firestore.js";

document.addEventListener('DOMContentLoaded', async function() {
    const votingContainer = document.querySelector('.voting-buttons');
    if (!votingContainer) return;
  
    const postId = votingContainer.dataset.postId;
    console.log("Post ID:", postId);
    const upvoteBtn = votingContainer.querySelector('.upvote');
    const downvoteBtn = votingContainer.querySelector('.downvote');
    const upvoteCount = votingContainer.querySelector('.upvote-count');
    const downvoteCount = votingContainer.querySelector('.downvote-count');
  
    // Wait for Firebase to be initialized
    while (!window.firebaseApp) {
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Get Firestore instance
    const db = getFirestore(window.firebaseApp);
    
    // Get votes document reference
    const votesRef = doc(db, 'votes', postId);
  
    // Track voted state
    let hasVoted = false;
    
    // Initialize or get votes
    async function initializeVotes() {
        try {
            const docSnap = await getDoc(votesRef);
            
            if (!docSnap.exists()) {
                // Document doesn't exist, create it
                await setDoc(votesRef, {
                    upvotes: 0,
                    downvotes: 0,
                    createdAt: serverTimestamp()
                });
                upvoteCount.textContent = '0';
                downvoteCount.textContent = '0';
            } else {
                // Document exists, get data
                const data = docSnap.data();
                upvoteCount.textContent = data.upvotes || 0;
                downvoteCount.textContent = data.downvotes || 0;
            }
            
            // Check if user has voted before
            hasVoted = localStorage.getItem(`voted_${postId}`);
            if (hasVoted === 'up') {
                upvoteBtn.classList.add('voted');
            } else if (hasVoted === 'down') {
                downvoteBtn.classList.add('voted');
            }
        } catch (error) {
            console.error("Error initializing votes:", error);
            upvoteCount.textContent = '0';
            downvoteCount.textContent = '0';
        }
    }

    await initializeVotes();
  
    // Handle upvote
    upvoteBtn.addEventListener('click', async () => {
        if (hasVoted) return;
        
        try {
            const docSnap = await getDoc(votesRef);
            if (!docSnap.exists()) {
                // Create document if it doesn't exist
                await setDoc(votesRef, {
                    upvotes: 1,
                    downvotes: 0,
                    lastUpdated: serverTimestamp()
                });
                upvoteCount.textContent = '1';
            } else {
                // Update existing document
                await updateDoc(votesRef, {
                    upvotes: increment(1),
                    lastUpdated: serverTimestamp()
                });
                
                const updated = await getDoc(votesRef);
                upvoteCount.textContent = updated.data().upvotes;
            }
            
            localStorage.setItem(`voted_${postId}`, 'up');
            upvoteBtn.classList.add('voted');
            hasVoted = 'up';
        } catch (error) {
            console.error("Error upvoting:", error);
        }
    });
  
    // Handle downvote
    downvoteBtn.addEventListener('click', async () => {
        if (hasVoted) return;
        
        try {
            const docSnap = await getDoc(votesRef);
            if (!docSnap.exists()) {
                // Create document if it doesn't exist
                await setDoc(votesRef, {
                    upvotes: 0,
                    downvotes: 1,
                    lastUpdated: serverTimestamp()
                });
                downvoteCount.textContent = '1';
            } else {
                // Update existing document
                await updateDoc(votesRef, {
                    downvotes: increment(1),
                    lastUpdated: serverTimestamp()
                });
                
                const updated = await getDoc(votesRef);
                downvoteCount.textContent = updated.data().downvotes;
            }
            
            localStorage.setItem(`voted_${postId}`, 'down');
            downvoteBtn.classList.add('voted');
            hasVoted = 'down';
        } catch (error) {
            console.error("Error downvoting:", error);
        }
    });
});